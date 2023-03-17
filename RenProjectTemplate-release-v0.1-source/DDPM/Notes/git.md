git导出代码的方法~archive


如果你用过svn，一定知道svn export，可以用来从代码库中导出一份干净的代码（没有.svn等）。git是否有类似功能呢？

git archieve
可以用于将库中代码打包。

1. 基本用法：
git archive --format tar.gz --output "./output.tar.gz" master
说明：

将master分支打包为output.tar.gz
--format指明打包格式，若不指明此项，则根据--output中的文件名推断文件格式。所以你也可以将上述命令简化为:
git archive --output "./output.tar.gz" master
可以使用-l参数获得支持的文件格式列表。

[@sjs_73_171 gittest]$ git archive -l
tar
tgz
tar.gz
zip
--output指明输出包名
2. 打包不同的分支或commit
如果想打包不同分支，只要使用不同分支名即可。比如我有一个分支名为“testbr”，可以用如下命令将其打包。

git archive --format tar.gz --output "./output.tar.gz" testbr
如果要打包某个commit，只要先使用git log找到相应的commit id, 再使用该commit id打包即可。比如：

git archive --format tar.gz --output "./output.tar.gz" 5ca16ac0d603603
3. 打包某个目录
如果想打包master下的mydir mydir2目录，可以用下面命令

git archive --format tar.gz --output "./output.tar.gz" master mydir mydir2  
4. 注意
打包建议在代码库的根目录下进行，不然会碰到各种问题。比如，如果在master分支的mydir目录下执行如下命令：

git archive --output "./output.tar.gz" master
就只会将mydir目录下的内容打包。



