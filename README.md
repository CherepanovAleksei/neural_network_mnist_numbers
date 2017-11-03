# Neural network for Mnist Numbers
Neural network for [MNIST database](http://yann.lecun.com/exdb/mnist/ "source") of handwritten digits(60000), that train and says is it handwritten number or not(noise).
Python3.6
Correctness of the algorithm is 99-100% on 10000 handwritten digits from MNIST and random noise.

## Number of layers:
| Number of layers | Testing numbers | Noise | Rectangle | Non-handwritten seven |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | 85% | 0% | 64% | 69% |
| 2 | 99-100% | 0-1% | 100% | 75% |
| 3 | 99-100% | 0% | 100% | 64% |

### Rectangle:
```
............................
............................
............................
............................
............................
............................
............................
............................
............................
............................
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
........@@@@@@@@@@@@........
............................
............................
............................
............................
............................
............................
............................
............................
............................
............................
```
### Non-handwritten seven:
```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
...........................@
..........................@.
.........................@..
........................@...
.......................@....
......................@.....
.....................@......
....................@.......
...................@........
..................@.........
.................@..........
................@...........
...............@............
..............@.............
.............@..............
............@...............
...........@................
..........@.................
.........@..................
........@...................
.......@....................
......@.....................
.....@......................
....@.......................
...@........................
..@.........................
.@..........................
```
## Example of right output:
```
1.0000000000
............................
............................
............................
............................
............................
............................
............................
............................
......@@@@@@................
...........@@@@@@@@@@.......
...................@@.......
...................@@.......
..................@@........
..................@@........
.................@@.........
.................@..........
................@@..........
................@...........
...............@@...........
..............@@............
.............@@@............
.............@@.............
............@@..............
............@@..............
...........@@@..............
...........@@@..............
...........@@...............
............................

0.9464285714
............................
............................
............................
.............@@.............
..........@@@@@@@...........
.........@@@@..@@...........
........@@@....@@...........
........@@.....@@...........
..............@@@...........
..............@@............
.............@@@............
.............@@.............
............@@..............
...........@@@..............
...........@@...............
..........@@................
..........@@................
.........@@.................
........@@@.................
........@@@.................
........@@@@@@@@...@@@@@@@..
.........@@@@@@@@@@@@@@.....
..............@@@...........
............................
............................
............................
............................
............................

1.0000000000
............................
............................
............................
............................
.................@..........
.................@..........
.................@..........
................@...........
................@...........
...............@@...........
...............@@...........
...............@............
...............@............
..............@@............
..............@.............
..............@.............
.............@@.............
.............@..............
.............@..............
............@@..............
............@@..............
............@@..............
...........@@...............
............................
............................
............................
............................
............................

1.0000000000
............................
............................
............................
............................
..............@@............
.............@@@............
.............@@@............
............@@@@............
..........@@@@@@@@@.........
..........@@@@@@@@@@........
.........@@@@@....@@........
........@@@@@.....@@@.......
........@@@........@@@......
........@@@.........@@......
........@@.........@@@......
........@@.........@@@......
.......@@@........@@@@......
.......@@@.......@@@@.......
.......@@@......@@@@........
.......@@@..@@@@@@@@........
........@@@@@@@@@@@.........
........@@@@@@@@@...........
..........@@@@@@............
............@...............
............................
............................
............................
............................

1.0000000000
............................
............................
............................
............................
............................
...........@................
...........@................
...........@........@.......
..........@.........@.......
.........@@.........@.......
.........@.........@@.......
........@@.........@........
........@.........@@........
.......@@.........@@........
.......@..........@.........
.......@@........@@.........
........@........@@.........
.........@@@@@@@@@@.........
.................@@.........
.................@@.........
.................@@.........
.................@@.........
.................@@.........
................@@..........
............................
............................
............................
............................

1.0000000000
............................
............................
............................
............................
............................
.................@..........
................@@@.........
................@@..........
...............@@@..........
...............@@@..........
...............@@...........
...............@@...........
..............@@@...........
..............@@............
..............@@............
.............@@@............
.............@@.............
.............@@.............
............@@@.............
............@@..............
............@@..............
............@@..............
...........@@...............
............@...............
............................
............................
............................
............................

0.9999999130
............................
............................
............................
............................
............................
............................
.........@@..........@@.....
.........@...........@......
........@@..........@.......
........@..........@@.......
.......@@..........@........
.......@..........@@........
.......@@........@@.........
.......@@@@@@@@@@@@.........
..........@@....@@..........
................@@..........
................@...........
...............@@...........
...............@............
..............@@............
..............@@............
.............@@.............
.............@@..@..........
.............@@@@...........
............................
............................
............................
............................

0.9999994375
............................
............................
............................
............................
............................
............................
............................
...........@@@..............
..........@@@@..............
..........@@@@@@............
.........@@@.@@@@...........
.........@....@@@@..........
.........@.....@@...........
.........@....@@@...........
.........@@...@@@@..........
.........@@@@@..@@..........
..........@@@....@@.........
..................@.........
..................@@........
...................@........
...................@@.......
....................@.......
....................@@......
.....................@......
......................@.....
......................@.....
............................
............................

0.9999945042
............................
............................
............................
............................
............................
.................@@@@@@@@...
..............@@@@@@@@@@@...
.............@@@@@@@@.......
.............@@@............
........@@..................
........@...................
.......@@...................
......@@....................
......@.....................
.....@@@....................
.....@@@@@..................
......@@@@@@@@@@@@..........
.......@@@@@@@@@@@@.........
............@@@@@@@.........
............@@...@@.........
............@@@.@@@.........
.............@@@@@..........
..............@@@...........
............................
............................
............................
............................
............................

1.0000000000
............................
............................
............................
............................
............................
............................
............................
................@...........
.............@@@@@@@@.......
...........@@@@..@.@@@......
.........@@@........@@@.....
........@@@......@..@@@.....
.......@@@........@@@@......
.......@@@.....@@@@@@@......
........@@@@@@@@@@@@........
...............@@@@.........
...............@@@..........
..............@@@...........
.............@@@@...........
............@@@@............
............@@@.............
...........@@@..............
..........@@@@..............
..........@@@...............
.........@@@................
.........@@@................
..........@.................
............................
```
## Another example:
```
0.0019937553
...@@..@....@.@...@.@@.@....
..@.@....@.@.@@.@@......@.@.
@.......@.@.@.@.@........@..
.@..........@@..@......@@..@
....@..@@..@.@.@.@.........@
@...@...@@..@.....@.........
@.....@...@.............@...
@.......@..@......@.....@@..
...@...................@.@@.
.@@..@@.....@...@..@.@......
...@.@..........@....@..@@..
.@.@.@...............@@@@...
@.............@...@...@.....
@..........@..@@..@.....@@@.
.......@....@..@.@......@...
.....@...@.....@....@...@..@
.....@@.@...@.@........@.@@.
@.@.@........@@.....@....@@@
..@@.@.@@....@.......@......
.........@..@.@..@..........
...............@.@.........@
.@@.....@.......@.@...@..@.@
..............@.....@.@....@
@@..@....@@@..@.@.........@.
.....@...@..@...@...@....@@.
......@...@@@..@.......@....
..@@.@......@..@@....@......
......@......@.@@.....@...@@

0.0019937553
...@.@..@.........@..@...@..
.@.........@@.......@.......
@@...@....@@@.....@@...@..@.
.@@.@@..@.@@.......@.....@..
.....@@@.......@...@........
@.@..@@@....@..@...@..@.....
@.......................@...
..........@.....@...@.......
.......@..@....@......@.@.@.
@@.....@@.@.@@....@@@.......
...@.@@....@@...........@...
.@....@....@@..@@.........@@
.@.@@.@@@.....@.........@@.@
..@.....@@....@@@.@@........
...@..@..@.@.....@.....@....
...@..@.@.@....@............
......@@..........@......@.@
.......@@@@.@@....@.........
@..........@...........@@@..
.@@@@.....@@@.@..@..@...@@..
.@....@@...@.@..@..@@.@@.@..
.........@............@.....
@@@..@.@........@@..@......@
...@@..@..@@..@....@..@...@.
.........@...@..@...@@@...@@
.....@...@...........@..@...
@...@....@.@@....@..@.@@....
...@.....@....@@.@......@.@@

0.0019937553
@..@..@.......@.....@@......
@...@.@.@..@.@......@.@@@..@
......@..@...@...@.@..@@....
@.....@@.@..@....@..@.......
...@@.@..@..@......@.@...@@.
...@.......@....@@@.........
.................@....@..@..
...@......@............@@.@.
@..@@..........@@@....@...@@
........@...........@.@..@..
...........@@..@........@...
....@....@@.@@@.......@@...@
@........@@.............@...
...@.@@.........@........@@@
.@.......@...@.....@..@....@
...@.@@..@..@....@..@.@@....
@.....@....@......@.@....@..
.@@.....@...@...@...@.@...@@
@...@@..................@...
@.@.........@@...@..@......@
.@...@.@......@@.@......@..@
...@....@.....@..........@@.
........@...@.@........@.@.@
....@.@@@@....@...........@.
.........@..@......@@@....@@
.......................@....
@@.....@.@@...@..........@.@
.....@.....@.@........@@.@..
```
