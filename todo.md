# states

## ideal world

- card with same number [0,9] 10
- card with same color [0,19] 20
- draw 2 with same color [0,2] 3
- wild card [0,4] 5
- draw 4 [0,4] 5
- skip card with same color [0,2] 3
- reverse card with same color [0,2] 3
- cards in P1 hand [0,108] 109
- cards in P2 hand [0,108] 109
- cards in P3 hand [0,108] 109
- last color drawn P1 [0,3] 4
- last number drawn P1 [0, 9] 10
- last color drawn P2 [0,3] 4
- last number drawn P2 [0, 9] 10
- last color drawn P3 [0,3] 4
- last number drawn P3 [0, 9] 10
- direction of play [0,1] 2

xx trillion possibilities => \inf training time?

## with caps

- card with same number [0,1] 2
- card with same color [0,2] 3 (excludes 'magic' cards)
- draw 2 with same color [0,1] 2
- wild card [0,1] 2
- draw 4 [0,1] 2
- skip card with same color [0,1] 2
- reverse card with same color [0,1] 2
- next / next next player has 1 card [0, 3] 4
- last color drawn P1 [0,3] 4
- last color drawn P2 [0,3] 4

* 7 actions

 2250 spaces => 5m training
84000 spaces => 3.1hrs training

# rewards

```python
rewards = {
    'play_card': 10,
    'two_left': 100,
    'uno': 500,
    'win': 10000,
    'lose': -10000,
}
```

# actions

```python
actions = {
    1: "match_color",
    2: "match_number",
    3: "skip",
    4: "reverse",
    5: "draw_2",
    6: "draw_4",
    7: "wild"
}
```
If player cannot play, draw 1 card

Choose most recently picked up color from one of the other players;
choose one it has the most of between the two if tied.
^^ Separate algo, for dealing with later.

# training

1. action space -> Rahul
2. hash function -> Frank

# adjustments from there
3. pickling -> next time !