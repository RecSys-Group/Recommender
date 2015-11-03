This is a mf methods. Users and items are represented as vectors. 

input:

1.item.txt: This file is the set of items
format:
---------
,0
iid(int),item(int)
---------
2.users.txt:This file is the set of users
format:
---------
,0
uid(int),user(int)
---------
3.new_ui.txt:This file is the users' transaction records, number_of_times indicate how many times the user purchased the item
format:
---------
,0,1,2
line,uid(float),iid(float),number_of_times(float)
---------
4.newtest.txt:This file is used to test topN 
format:
---------
,0,1,...
uid(int),item1,item2,...
---------
5.new_col_nor_ui_01.txt:This file is nn normalized user representation file
format:
---------
,0,1,2
line,uid(float),uid(float),value(float)
---------
6.nor_seq_simi_file:This file is seq normalized user representation file
format:
---------
,0
0,"{'1463@6034': 1.0, '1405@17': 0.7071067811865475, ...}"
---------
7.new_format_seq:This file is seq formatted users' transaction records
format:
---------
7929 3053 3699 5842 2251 1405 1463 428 3003 48 4307(iid) -1 0 1 2 3 4 5 -1 421 5435 1577 511 3759 41 7481 1790 2491 3803 7886 1830 3978 2467 2545 4673 17 1367 6034 -1 272 421 3820 -1 -2
---------
8.users_of_pattern.txt: Users's patterns
format:
---------
,0,1
0,0@577,"['291', '1315', '1411', '456', '7980', '1005', '1521', '5365', '3064', '1177', '1051', '1438', '671']"
---------
9.tafeng_nor_SEQ_similar_100,tafeng_nor_NN_similar:nn and seq similar generate by gen_user_similarity.py which are the similarity based on sp and nsp.
format:
---------
,0,1,2,...
0,{'0': '1.0'},{'862': '0.236590190256'},{'5395': '0.163744946156'},...
---------
