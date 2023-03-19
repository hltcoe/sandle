#!/usr/bin/env python3

import sys
assert len(sys.argv) == 2
token = sys.argv[1]
print('{"app": {}}')
print(f'Authorization: Bearer {token}')
