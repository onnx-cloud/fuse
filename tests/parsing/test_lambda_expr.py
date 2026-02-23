from src import parser


def test_arrow_lambda_parses():
    p = parser._FuseParserWrapper(parser.GRAMMAR)
    src = """
node foo() {
  x = Loop< body=(i, s) => (true, Add(s, <f32>(i))), v_initial=[0] >(10, true)
}
"""
    ast = p.parse(src)

    # find lambda nodes
    def find_lambdas(x):
        if isinstance(x, dict):
            if "lambda" in x:
                return [x]
            out = []
            for v in x.values():
                out.extend(find_lambdas(v))
            return out
        if isinstance(x, list):
            out = []
            for i in x:
                out.extend(find_lambdas(i))
            return out
        return []

    lambdas = find_lambdas(ast)
    assert len(lambdas) >= 1
    ln = lambdas[0]
    assert ln["lambda"]["args"] == ["i", "s"]
    # body should be tuple-like with cond and next-state
    assert isinstance(ln["lambda"]["body"], list)
    assert ln["lambda"]["body"][0] is True
