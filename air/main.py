from air.processing.preprocessing import PreProcessor

# from air.data.load import load_raw_data
from air.data.load import load_raw_data

preprocessor = PreProcessor()


def main():
    data = load_raw_data()
    res = preprocessor.process(data)
    print(res.head())


if __name__ == "__main__":
    main()
