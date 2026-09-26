import socket

HOST = "its-live-data.s3.amazonaws.com"

print("Resolving:", HOST)

try:
    addresses = socket.getaddrinfo(HOST, 443)

    for address in addresses:
        print(address)

except Exception as e:
    print(type(e).__name__, e)