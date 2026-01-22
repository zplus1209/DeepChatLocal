import uuid

def make_anchor():
    return str(uuid.uuid4())

def md_with_anchor(text: str):
    anchor = make_anchor()
    output = f"""<a id='{anchor}'></a
## {text}
"""
    return anchor, output