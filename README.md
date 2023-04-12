
# Scriptaly App

This is a Streamlit app for loading, modifying, and downloading a file. It allows the user to perform the following actions:
- Load a CSV file or other types 
- Modify the data types of the columns and replace specific values
- Download the modified CSV file
- Generate the Python code used to modify the CSV file

## Usage

To run the app, execute the following command in your terminal:

```streamlit run app.py```

## Features

### Load Data

To load your CSV file, click on the "Load Data" button in the sidebar and select your CSV file. The data will be displayed in an expander box.

### Data Type Modification and Value Replacement

To modify the data types of the columns and replace specific values, click on the "Type Modification and Value Replacement" button in the sidebar. A form will appear where you can choose the columns and set the new data types and values to replace.

### Download Data

To download the modified CSV file, click on the "Download Data" button in the sidebar. The modified CSV file will be downloaded to your local machine.

### Code Snippet

To generate the Python code used to modify the CSV file, scroll down to the bottom of the page and you will find the code snippet.

## Notes

- The modified data is stored in the cache, so if you refresh the page or close the app, the modified data will be lost.
- You can use the "Delete all the data stored in the cache" button to clear the cache.
- The app hides Streamlit's default header, footer, and menu styles.

