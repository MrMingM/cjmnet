"""Bounded HTTPS probes and resumable downloads; signed URLs never leave memory."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from types import SimpleNamespace
import urllib.error
import urllib.request
from urllib.parse import urlsplit, unquote
import zipfile

from scorw_repair.core import (Blocked, atomic_json, atomic_bytes, read_json, digest, identity,
                               lock, checked_path)

ALLOWED = re.compile(r"^(lidar_hdl64_last\.(?:zip|z\d{2,3}|zip\.\d{3})|weather_station\.zip|labeltool_labels\.zip|calib_tf_tree_full\.json|calib_cam_stereo_left\.json|cam_left_labels_TMP\.zip)$")


def archive_tool():
    for name in ("/data/cjm/tools/7zip-26.04/bin/7zz",shutil.which("7zz"),shutil.which("7z")):
        if name and Path(name).is_file():
            return name
    return None


def urls(path):
    path = Path(path)
    if not path.is_file():
        raise Blocked("BLOCKED_URL_LIST_MISSING: private stf_urls.txt")
    # No file digest or URL/query is written into a public contract.
    if os.name == "posix":
        path.chmod(0o600)
    result = {}
    for line in path.read_text("utf-8-sig").splitlines():
        value = line.strip()
        # Also accept the user's existing Markdown-wrapped list without printing it.
        if not value.startswith("https://"):
            continue
        u = urlsplit(value)
        name = unquote(u.path.rsplit("/", 1)[-1])
        if u.username or u.password or not u.hostname:
            raise Blocked("BLOCKED_URL_FORMAT: userinfo URLs are forbidden")
        if not ALLOWED.fullmatch(name):
            continue  # Never download unrelated cameras, gates, CAN or other sensors.
        if name in result:
            raise Blocked("BLOCKED_DUPLICATE_FILENAME: " + name)
        result[name] = value
    if not result:
        raise Blocked("BLOCKED_URL_LIST_EMPTY: no allowed HTTPS entries")
    return result


class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise Blocked("BLOCKED_INSECURE_REDIRECT")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(url, method="GET", headers=None):
    try:
        if urlsplit(url).scheme != "https":
            raise Blocked("HTTPS_REQUIRED")
        return urllib.request.build_opener(HTTPSOnly()).open(
            urllib.request.Request(url, method=method, headers=dict(headers or {}, **{"Accept-Encoding":"identity"})), timeout=40)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        raise Blocked("REMOTE_HTTP_" + str(code)) from None
    except Blocked:
        raise
    except Exception:
        # Exception text may contain a signed URL, redirect target or credential.
        raise Blocked("REMOTE_TRANSPORT_ERROR") from None


def probe(url):
    size = None
    try:
        with request(url, "HEAD") as reply:
            if reply.headers.get("Content-Length"):
                size = int(reply.headers["Content-Length"])
    except (Blocked, ValueError):
        pass  # HEAD rejection never establishes that the GET link is invalid.
    with request(url, headers={"Range":"bytes=0-3"}) as reply:
        status = reply.status
        head = reply.read(4)
        content_range = reply.headers.get("Content-Range", "")
        if status == 206:
            match = re.fullmatch(r"bytes 0-\d+/(\d+)", content_range)
            if not match:
                raise Blocked("BLOCKED_INVALID_CONTENT_RANGE")
            total = int(match.group(1))
            if size is not None and size != total:
                raise Blocked("BLOCKED_REMOTE_SIZE_CHANGED")
            size = total
        elif status == 200:
            if reply.headers.get("Content-Length"):
                size = int(reply.headers["Content-Length"])
        else:
            raise Blocked("BLOCKED_GET_PROBE_STATUS")
        validator = reply.headers.get("ETag") or reply.headers.get("Last-Modified")
    if not size or size < 1:
        raise Blocked("BLOCKED_REMOTE_SIZE_UNKNOWN")
    return dict(bytes=size, range_supported=status == 206,
                validator_sha256=identity(validator) if validator else None,
                signature_hex=head.hex())


def read_range(url, start, stop, total):
    if not 0 <= start <= stop < total or stop-start+1 > 131072:
        raise Blocked("BLOCKED_PROBE_BUDGET")
    with request(url, headers={"Range":f"bytes={start}-{stop}"}) as reply:
        expected = f"bytes {start}-{stop}/{total}"
        if reply.status != 206 or reply.headers.get("Content-Range") != expected:
            raise Blocked("BLOCKED_RANGE_UNSUPPORTED_FOR_ARCHIVE_INSPECTION")
        data = reply.read(stop-start+2)
        if len(data) != stop-start+1:
            raise Blocked("BLOCKED_SHORT_RANGE")
        return data


def zip_layout(tail, total):
    """EOCD/ZIP64 locator proves container disk count, not whole dataset coverage."""
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or len(tail)-at < 22:
        raise Blocked("BLOCKED_ZIP_DIRECTORY_MISSING")
    disk, start_disk, entries_disk, entries, directory_bytes, offset, comment = struct.unpack_from("<4H2IH", tail, at+4)
    if at+22+comment != len(tail):
        raise Blocked("BLOCKED_ZIP_TRAILING_OR_TRUNCATED_BYTES")
    volumes = disk+1
    if at >= 20 and tail[at-20:at-16] == b"PK\x06\x07":
        locator_disk, zip64_offset, volumes = struct.unpack_from("<IQI", tail, at-16)
        if volumes < 1:
            raise Blocked("BLOCKED_INVALID_ZIP64_VOLUMES")
    elif disk == 65535 or start_disk == 65535:
        raise Blocked("BLOCKED_ZIP64_LOCATOR_MISSING")
    if start_disk > disk and disk != 65535:
        raise Blocked("BLOCKED_ZIP_DISK_IDENTITY")
    if disk != 65535 and volumes != disk+1:
        raise Blocked("BLOCKED_ZIP_VOLUME_COUNT_INCONSISTENT")
    return dict(container_volumes=volumes, last_disk=disk, central_directory_start_disk=start_disk,
        zip64=entries == 65535 or offset == 0xffffffff or directory_bytes == 0xffffffff,
        required_names=[f"lidar_hdl64_last.z{i:02}" for i in range(1,volumes)] + ["lidar_hdl64_last.zip"],
        dataset_coverage="not established by EOCD; frozen sample-member coverage checked after archive listing")


def official_checksums(config):
    path = Path(config["stf_code"])/"SeeingThroughFog_sha256sum.txt"
    result = {}
    for line in path.read_text("utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch("[0-9a-f]{64}", parts[0]):
            result[parts[1].lstrip("*")] = parts[0]
    return result


def preflight(config):
    links = urls(config["private_urls"])
    rows, problems = {}, []
    for name, link in sorted(links.items()):
        try:
            rows[name] = dict(status="reachable", **probe(link))
        except Blocked as error:
            rows[name] = dict(status=str(error))
            problems.append(dict(filename=name, status=str(error)))
        except Exception:
            rows[name] = dict(status="REMOTE_PROBE_READ_ERROR")
            problems.append(dict(filename=name,status="REMOTE_PROBE_READ_ERROR"))
    required = ["lidar_hdl64_last.zip", "weather_station.zip", "calib_tf_tree_full.json"]
    missing = sorted(set(required)-set(links))
    layout = None
    if rows.get("lidar_hdl64_last.zip", {}).get("status") == "reachable":
        size = rows["lidar_hdl64_last.zip"]["bytes"]
        try:
            tail = read_range(links["lidar_hdl64_last.zip"], max(0,size-65577), size-1,size)
            layout = zip_layout(tail, size)
            missing += sorted(set(layout["required_names"])-set(links))
        except Blocked as error:
            problems.append(dict(filename="lidar_hdl64_last.zip",status=str(error)))
        except Exception:
            problems.append(dict(filename="lidar_hdl64_last.zip",status="BLOCKED_ARCHIVE_PROBE_READ_ERROR"))
    base = Path(config["reference_root"])
    ancestor = next(p for p in (base,*base.parents) if p.exists())
    free = shutil.disk_usage(ancestor).free
    total = sum(r.get("bytes",0) for r in rows.values())
    # Include the declared cap for selective extraction and a disk reserve.
    needed = total + config["download"]["extract_budget_bytes"] + config["download"]["reserve_bytes"]
    # Existing verified archives reduce required new download space only.
    for name,r in rows.items():
        p = base/"archives"/name
        if p.is_file() and p.stat().st_size == r.get("bytes"):
            needed -= p.stat().st_size
        else:
            partial=p.with_suffix(p.suffix+".partial");state=p.with_suffix(p.suffix+".identity.json")
            if partial.is_file() and state.is_file() and r.get("validator_sha256"):
                if (read_json(state).get("remote")=={k:r[k] for k in ("bytes","validator_sha256")}
                        and partial.stat().st_size<=r["bytes"]):
                    needed-=partial.stat().st_size
    if total > config["download"]["maximum_download_bytes"]:
        problems.append(dict(status="BLOCKED_DOWNLOAD_BUDGET"))
    if free < needed:
        problems.append(dict(status="BLOCKED_INSUFFICIENT_SPACE"))
    if layout and layout["container_volumes"] > 1 and not archive_tool():
        problems.append(dict(status="BLOCKED_SPLIT_ZIP_TOOL_MISSING"))
    if missing:
        status = "BLOCKED_MISSING_VOLUMES" if any("lidar_hdl64_last" in n for n in missing) else "BLOCKED_MISSING_FILES"
    else:
        status = "BLOCKED_PREFLIGHT" if problems else "ready"
    return dict(status=status,files=rows,missing=sorted(set(missing)),problems=problems,layout=layout,
        free_bytes=free,required_free_bytes=max(0,needed),download_bytes=total,
        optional_missing=sorted({"labeltool_labels.zip","calib_cam_stereo_left.json","cam_left_labels_TMP.zip"}-set(links)),
        historical_checksum_scope="bundled official checksum list describes old SeeingThroughFogCompressed.z01..z18+zip; per-sensor names require exact-name official checksum or local-hash-only status",
        claimed_sensor_volume_count=config["download"]["claimed_sensor_volumes"],
        volume_count_policy="use inspected EOCD/ZIP64 count; never infer six missing URLs from an unverified claim")


def fetch(url, row, destination, expected_sha=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    state = destination.with_suffix(destination.suffix+".identity.json")
    contract = {k:row[k] for k in ("bytes","validator_sha256")}
    part = destination.with_suffix(destination.suffix+".partial")
    if any(p.is_symlink() for p in (destination,state,part)):
        raise Blocked("BLOCKED_ARCHIVE_SYMLINK")
    if state.exists() and read_json(state)["remote"] != contract:
        raise Blocked("BLOCKED_REMOTE_IDENTITY_CHANGED: " + destination.name)
    if destination.exists():
        if not state.exists():
            raise Blocked("BLOCKED_UNREGISTERED_ARCHIVE: " + destination.name)
        saved = read_json(state)
        if destination.stat().st_size != row["bytes"] or digest(destination) != saved.get("sha256"):
            raise Blocked("BLOCKED_LOCAL_ARCHIVE_CHANGED: " + destination.name)
        if expected_sha and saved.get("sha256")!=expected_sha:
            raise Blocked("BLOCKED_OFFICIAL_SHA256_MISMATCH: " + destination.name)
        return saved
    if part.exists() and (not state.exists() or not row["validator_sha256"]):
        raise Blocked("BLOCKED_UNVERIFIABLE_PARTIAL: " + destination.name)
    atomic_json(state,dict(remote=contract,status="downloading"))
    start = part.stat().st_size if part.exists() else 0
    if start > row["bytes"]:
        raise Blocked("BLOCKED_OVERSIZED_PARTIAL: " + destination.name)
    if start < row["bytes"]:
        with request(url,headers={"Range":f"bytes={start}-"} if start else {}) as reply:
            if start and (reply.status != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{row['bytes']-1}/{row['bytes']}"):
                raise Blocked("BLOCKED_RESUME_RANGE_REJECTED: " + destination.name)
            validator = reply.headers.get("ETag") or reply.headers.get("Last-Modified")
            if row["validator_sha256"] and identity(validator) != row["validator_sha256"]:
                raise Blocked("BLOCKED_DOWNLOAD_REMOTE_CHANGED: " + destination.name)
            with part.open("ab" if start else "wb") as stream:
                count = start
                last_log=count
                try:
                    while chunk := reply.read(1024*1024):
                        count += len(chunk)
                        if count > row["bytes"]:
                            raise Blocked("BLOCKED_DOWNLOAD_LENGTH_EXCEEDED")
                        stream.write(chunk)
                        if count-last_log>=512*1024*1024:
                            print(f"DOWNLOAD {destination.name}: {count}/{row['bytes']} bytes",flush=True)
                            last_log=count
                    stream.flush(); os.fsync(stream.fileno())
                except Blocked:
                    raise
                except Exception:
                    raise Blocked("DOWNLOAD_INTERRUPTED: " + destination.name) from None
    if part.stat().st_size != row["bytes"]:
        raise Blocked("DOWNLOAD_INCOMPLETE: " + destination.name)
    actual = digest(part)
    if expected_sha and actual != expected_sha:
        raise Blocked("BLOCKED_OFFICIAL_SHA256_MISMATCH: " + destination.name)
    os.replace(part,destination)
    result = dict(remote=contract,status="downloaded",sha256=actual,official_sha256=expected_sha,
                  authenticity="official_checksum_match" if expected_sha else "local_hash_only_not_official_authentication")
    atomic_json(state,result)
    return result


def download_all(config, report, allowed):
    if report["status"] != "ready":
        raise Blocked(report["status"] + ": " + ", ".join(report["missing"]))
    links = urls(config["private_urls"])
    checksums = official_checksums(config)
    base = Path(config["reference_root"])/"archives"
    records = {}
    with lock(base):
        for name,row in report["files"].items():
            if not allowed and not (base/name).exists():
                raise Blocked("BLOCKED_DOWNLOAD_DISABLED: " + name)
            print(f"DOWNLOAD/VERIFY {name}: {row['bytes']} bytes",flush=True)
            records[name] = fetch(links[name],row,base/name,checksums.get(name))
            print("VERIFIED "+name+": "+records[name]["authenticity"],flush=True)
    return records


class SplitZipReader:
    """Use the existing official 7z extraction route, streaming selected members only."""
    def __init__(self,path):
        self.path,self.tool = str(path),archive_tool()
        if not self.tool:
            raise Blocked("BLOCKED_SPLIT_ZIP_TOOL_MISSING")
        listed=subprocess.run([self.tool,"l","-slt","-ba",self.path],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        if listed.returncode or len(listed.stdout)>32*1024*1024:
            raise Blocked("BLOCKED_SPLIT_ZIP_DIRECTORY")
        self.entries=[]
        for block in listed.stdout.decode("utf-8",errors="strict").replace("\r\n","\n").split("\n\n"):
            row=dict(line.split(" = ",1) for line in block.splitlines() if " = " in line)
            if "Path" not in row or "Size" not in row:
                continue
            if "L" in row.get("Attributes","") or row.get("Encrypted")=="+":
                raise Blocked("BLOCKED_ARCHIVE_LINK_OR_ENCRYPTION")
            self.entries.append(SimpleNamespace(filename=row["Path"],file_size=int(row["Size"]),
                flag_bits=0,external_attr=0,is_dir=lambda folder=row.get("Folder"):folder=="+"))
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def infolist(self): return self.entries
    def testzip(self):
        result=subprocess.run([self.tool,"t","-bd","-bb0",self.path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if result.returncode:
            raise Blocked("BLOCKED_SPLIT_ZIP_INTEGRITY")
        return None
    def read(self,info):
        process=subprocess.Popen([self.tool,"x","-so","-spd","-bd","-y",self.path,info.filename],
                                 stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        try:
            data=process.stdout.read(info.file_size+1)
            if len(data)!=info.file_size:
                raise Blocked("BLOCKED_ARCHIVE_MEMBER_LENGTH")
            if process.wait()!=0:
                raise Blocked("BLOCKED_ARCHIVE_MEMBER_CRC")
            return data
        finally:
            if process.poll() is None:
                process.kill();process.wait()
            process.stdout.close()


def selective_extract(config, samples):
    """CRC-check archives, then extract only the preregistered member whitelist."""
    base = Path(config["reference_root"])
    archive = base/"archives/lidar_hdl64_last.zip"
    wanted = {f"lidar_hdl64_last/{s['sample']}.bin" for s in samples}
    wanted_stems = {Path(n).stem for n in wanted}
    records, budget = [], 0
    for path in sorted((base/"archives").glob("*")):
        if path.suffix == ".json":
            if path.name in {"calib_tf_tree_full.json","calib_cam_stereo_left.json"}:
                data = path.read_bytes(); atomic_bytes(base/"selected"/path.name,data)
                records.append(dict(member=path.name,sha256=digest(path),bytes=len(data)))
            continue
        if path.suffix != ".zip":
            continue
        if path.name=="lidar_hdl64_last.zip" and (base/"archives/lidar_hdl64_last.z01").exists():
            package=SplitZipReader(path)
        else:
            try:
                package = zipfile.ZipFile(path)
            except (zipfile.BadZipFile, NotImplementedError):
                raise Blocked("BLOCKED_ARCHIVE_READER_UNSUPPORTED: " + path.name) from None
        with package:
            print("CRC checking "+path.name,flush=True)
            # Full CRC scan is deliberate after full authorized download; no full extraction.
            try:
                bad = package.testzip()
            except Exception:
                raise Blocked("BLOCKED_ARCHIVE_CRC: " + path.name) from None
            if bad is not None:
                raise Blocked("BLOCKED_ARCHIVE_CRC: " + path.name)
            found = set()
            for info in package.infolist():
                p = Path(info.filename)
                if info.is_dir():
                    continue
                sensor = next((s for s in ("lidar_hdl64_last","weather_station","labeltool_labels","cam_left_labels_TMP") if s in p.parts),None)
                # Some sensor archives contain bare filenames; archive name fixes identity.
                if sensor is None:
                    sensor = path.stem if path.stem in {"lidar_hdl64_last","weather_station","labeltool_labels","cam_left_labels_TMP"} else None
                if sensor is None or p.stem not in wanted_stems:
                    continue
                suffix = ".bin" if sensor == "lidar_hdl64_last" else ".txt" if sensor == "cam_left_labels_TMP" else ".json"
                if p.suffix != suffix:
                    continue
                key = f"{sensor}/{p.name}"
                if key in found or info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise Blocked("BLOCKED_ARCHIVE_MEMBER_IDENTITY")
                found.add(key)
                budget += info.file_size
                if budget > config["download"]["extract_budget_bytes"] or info.file_size > config["download"]["maximum_member_bytes"]:
                    raise Blocked("BLOCKED_SELECTIVE_EXTRACTION_BUDGET")
                destination = checked_path(base/"selected",key)
                try:
                    payload = package.read(info)
                except Exception:
                    raise Blocked("BLOCKED_SELECTED_MEMBER_CRC") from None
                atomic_bytes(destination,payload)
                records.append(dict(member=key,bytes=len(payload),sha256=digest(destination),archive=path.name))
    present = {r["member"] for r in records}
    missing = sorted(wanted-present)
    if missing:
        raise Blocked("BLOCKED_FROZEN_SAMPLE_MEMBERS_MISSING: " + ", ".join(missing))
    return dict(status="selected_members_verified",files=records,total_uncompressed_bytes=budget,
                all_sensors_extracted=False)
