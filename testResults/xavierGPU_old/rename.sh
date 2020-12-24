for file in *.log
do
  mv "${file}" "${file/xavier/}"
done

