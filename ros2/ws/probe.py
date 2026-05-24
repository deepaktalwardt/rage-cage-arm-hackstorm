"""Raw WebSocket probe of foxglove_bridge — bypasses python-websockets.

Connects to ws://localhost:8765, negotiates the foxglove.websocket.v1
subprotocol, and dumps channel announcements for /robot_description,
/joint_states, /tf, and /arm_status — including the schema body length
and head, so we can see whether disable_load_message took effect.
"""
import base64
import hashlib
import json
import os
import socket
import struct
import sys


HOST = "127.0.0.1"
PORT = 8765
WANTED = {"/robot_description", "/joint_states", "/tf", "/arm_status"}


def handshake(sock):
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"sec-websocket-protocol: foxglove.websocket.v1\r\n"
        f"\r\n"
    )
    sock.sendall(req.encode("ascii"))
    sock.settimeout(5)
    buf = b""
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
        if b"\r\n\r\n" in buf and len(buf) > 200:
            break
    headers, _, leftover = buf.partition(b"\r\n\r\n")
    print(headers.decode("ascii", "replace"), flush=True)
    print("--- body / leftover (first 400 bytes) ---", flush=True)
    print(leftover[:400].decode("ascii", "replace"), flush=True)
    return leftover


def read_frame(sock, leftover_holder):
    """Decode a single (text) WS frame, returning the payload as bytes.
       leftover_holder is a single-element list used as mutable buffer."""
    def recv_n(n):
        buf = leftover_holder[0]
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise RuntimeError("closed")
            buf += chunk
        leftover_holder[0] = buf[n:]
        return buf[:n]

    hdr = recv_n(2)
    b1, b2 = hdr[0], hdr[1]
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack(">H", recv_n(2))
    elif length == 127:
        (length,) = struct.unpack(">Q", recv_n(8))
    if masked:
        recv_n(4)  # not expected from server
    payload = recv_n(length) if length else b""
    return opcode, payload


def main():
    sock = socket.create_connection((HOST, PORT), timeout=10)
    leftover = handshake(sock)
    holder = [leftover]
    sock.settimeout(5)
    for _ in range(80):
        try:
            opcode, payload = read_frame(sock, holder)
        except (socket.timeout, RuntimeError) as e:
            print("done:", e, flush=True)
            break
        if opcode != 0x1:  # not text frame
            continue
        try:
            d = json.loads(payload.decode("utf-8"))
        except Exception:
            continue
        op = d.get("op")
        if op == "serverInfo":
            print("serverInfo capabilities=", d.get("capabilities"), flush=True)
            print("serverInfo supportedEncodings=", d.get("supportedEncodings"), flush=True)
        elif op == "advertise":
            for ch in d.get("channels", []):
                if ch["topic"] in WANTED:
                    sch = ch.get("schema", "")
                    print(
                        f"ch{ch['id']:>3}  {ch['topic']:<28} schemaName={ch.get('schemaName'):<28}"
                        f" encoding={ch.get('encoding'):<6} schemaEncoding={ch.get('schemaEncoding'):<10}"
                        f" schemaLen={len(sch)}",
                        flush=True,
                    )
                    if sch:
                        head = sch[:160].replace("\n", "\\n")
                        print("    schema head:", repr(head), flush=True)


if __name__ == "__main__":
    main()
