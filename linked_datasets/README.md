
# Linked datasets Folder

link your datasets via command below

```sh
ln -s $SOURCE_DIR $LINK
```

```ps1
New-Item -ItemType SymbolicLink -Path $LINK -Target $SOURCE_DIR
```
