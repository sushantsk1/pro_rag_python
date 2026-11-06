import json
import sys

from .parser import parse_document

if __name__ == "__main__":
    try:
        result = parse_document(sys.argv[1], sys.argv[2], sys.argv[3] == "True")
    except Exception as error:
        result = {
            "error": {
                "code": getattr(error, "code", "PARSE_ERROR"),
                "message": str(error)[:220] or "Document parsing failed.",
            }
        }
    sys.stdout.buffer.write(json.dumps(result).encode("utf-8"))
