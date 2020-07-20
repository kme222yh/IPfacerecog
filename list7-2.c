

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
