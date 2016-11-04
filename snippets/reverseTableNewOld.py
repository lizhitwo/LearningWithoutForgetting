import sys

def reverse(line):
    cells = line.split('&');
    assert(len(cells)%2 == 1)
    cells = cells[0:1] + [ cells[x*2+y+1] for x in range(int((len(cells)-1) / 2)) for y in [1,0] ]
    line = '&'.join(cells)
    return line

if __name__ == '__main__':
    lines = []
    while True:
        lines.append(sys.stdin.readline())
        if len(lines[-1]) == 0:
            break
        if lines[-1] == '\n':
            for line in lines[:-1]:
                linecontent = line.split('\\\\')
                if len(linecontent) == 2:
                    print( reverse(linecontent[0]) + '\\\\' + linecontent[1], end="", flush=True )
                else:
                    assert(line[-1] == '\n')
                    print( line[:-1] + ' % not changed\n', end="", flush=True )
            print()
            lines = []
