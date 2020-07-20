/*
    学籍番号 : 11870273
    名前     : 宮本 拓実
    課題     : list 7-2
    作成日   : 2018/
*/

#include<stdio.h>
#include<limits.h>

int main(void){
  printf("この処理系のchar型は");

  if(CHAR_MIN)
   puts("符号付き型です。");
  else
   puts("符号無し型です。");
  return 0;
}
