#!/usr/bin/env python3

token = 'dGVzdA=='  # base-64 encoding of "test"
print('{"app": {}}')
print(f'Authorization: Bearer {token}')
