import sys, contextlib

@contextlib.contextmanager
def suppress_print():
    save_stdout = sys.stdout
    sys.stdout = open('trash','w')
    yield
    sys.stdout = save_stdout

def test_print(print_line):
	print(print_line)

test_print('test')

with suppress_print():
	test_print('test')

