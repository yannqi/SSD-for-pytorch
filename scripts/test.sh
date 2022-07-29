#!/bin/bash
clear
 
echo "请输入你想查询文件夹大小大于多少g："
read variable
echo "大于$variable g的文件目录如下："

let temp=$[$variable*1024*1024]
# 默认文件深度为2，可指定修改
for i in $(du  --max-depth=2 | sort -n -r | awk '{print $2}' )
do  
    size=$(du  -s $i | sort -n -r | awk '{print $1}' )
    if [ $size -gt $temp ]; then
    size=$[${size}/1024/1024]
    echo '目录名：'${i} '文件大小：'$size'g'
    fi    
done

echo "查询完成" 