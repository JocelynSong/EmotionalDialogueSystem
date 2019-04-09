import xlrd


def main():
    file_path = "E:\\NLP\\code\\dialogue\\external emotion vocabulary\\emotional_vocab\\emotion_dict.xls"
    data = xlrd.open_workbook(file_path)

    table = data.sheet_by_name("Sheet1")
    nrows = table.nrows
    ncolums = table.ncols
    print("row=%d, colum=%d\n" % (nrows, ncolums))

    emotion_dict = dict()
    for i in range(1, 1364):
        word = table.cell(i, 1).value
        emotion = table.cell(i, 2).value
        if emotion not in emotion_dict.keys():
            emotion_dict[emotion] = list()
        emotion_dict[emotion].append(word)

    for emotion in emotion_dict.keys():
        file_name = emotion + ".word.txt"
        f = open(file_name, "w", encoding="utf-8")
        for word in emotion_dict[emotion]:
            f.write(word)
            f.write("\n")
        f.close()


if __name__ == "__main__":
    main()

