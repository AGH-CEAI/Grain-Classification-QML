import preprocessing


def main():
    print("Hello, world!")
    df = preprocessing.get_excel_data()
    df = preprocessing.drop_columns(df)
    df_2 = preprocessing.add_indirect_features(df)
    print(df.columns)


if __name__ == "__main__":
    main()
