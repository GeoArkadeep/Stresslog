import toga
from toga.style import Pack
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json
import os
import pandas as pd
import numpy as np

def custom_edit(app, expected_headers, expected_units, filename, port):
    webview = toga.WebView(style=Pack(flex=1))
    start_server(webview, expected_headers, expected_units, filename, port)
    webview.url = f'http://localhost:{port}/csv-editor'
    window = toga.Window(id='csv_editor', title='CSV Editor', size=(800, 600))
    window.content = webview
    app.windows.add(window)
    window.show()

def start_server(webview, expected_headers, expected_units, filename, port):
    def run_server():
        handler = lambda *args, **kwargs: create_csv_editor_handler(args, expected_headers, expected_units, filename, **kwargs)
        server = HTTPServer(('localhost', port), handler)
        print(f'Starting server on port {port}...')
        server.serve_forever()

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

def create_csv_editor_handler(args, expected_headers, expected_units, filename):
    class CSVEditorHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/csv-editor':
                print("bitchin")
                # Load or create the CSV file
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                else:
                    df = pd.DataFrame(columns=expected_headers)
                    df.to_csv(filename, index=False)

                # Convert DataFrame to list of lists and handle NaN values
                csv_data = [df.columns.tolist()] + df.replace({np.nan: None}).values.tolist()
                print(csv_data)

                # Replace placeholders with actual data
                filled_html = template.replace('{{ expectedHeaders }}', json.dumps(expected_headers))
                filled_html = filled_html.replace('{{ defaultUnits }}', json.dumps(expected_units))
                filled_html = filled_html.replace('{{ initialCSVData }}', json.dumps(csv_data))

                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(filled_html.encode('utf-8'))
            else:
                # Serve static files for other requests
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == '/save':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                # Parse the JSON data
                data = json.loads(post_data.decode('utf-8'))
                mapped_data = data['data']

                # Convert to a pandas DataFrame
                df = pd.DataFrame(mapped_data, columns=self.expected_headers)

                # Set up pint's unit registry
                ureg = UnitRegistry()
                
                # Convert units using pint
                for i, header in enumerate(df.columns):
                    target_unit = self.expected_units[i]

                    if target_unit == " ":
                        # For string columns, do nothing
                        continue
                    else:
                        # For numeric columns, attempt to convert units
                        df[header] = df[header].apply(lambda x: convert_value(x, target_unit, ureg))

                try:
                    df.to_csv(self.filename, index=False)
                    self.send_response(200)
                except Exception as e:
                    print(f"Error saving data: {e}")
                    self.send_response(500)
                self.end_headers()

            elif self.path == '/clear':
                try:
                    if os.path.exists(self.filename):
                        os.remove(self.filename)
                    self.send_response(200)
                except Exception as e:
                    print(f"Error clearing data: {e}")
                    self.send_response(500)
                self.end_headers()

    return CSVEditorHandler

def convert_value(value, target_unit, ureg):
    try:
        # Try to extract numeric value and unit
        match = re.match(r'([-+]?[0-9]*\.?[0-9]+)\s*([a-zA-Z]*)', str(value))
        if match:
            num, unit = match.groups()
            if unit:
                original_value = float(num) * ureg(unit)
                return f"{original_value.to(target_unit).magnitude:.6f}"
            else:
                # If no unit provided, assume it's already in the target unit
                return f"{float(num):.6f}"
        else:
            # If no match, return the original value
            return value
    except Exception as e:
        print(f"Error converting {value} to {target_unit}: {e}")
        return value  # Return original value if conversion fails



template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="HandheldFriendly" content="true">
    <title>CSV Mapping and Editor with Units</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .mapping-row { 
            display: flex;
            align-items: center;
            margin-bottom: 10px; 
        }
        .mapping-row label {
            width: 100px;
            margin-right: 10px;
        }
        select { width: 200px; }
        button { margin-top: 20px; margin-right: 10px; }
        #csvInfo { margin-top: 20px; }
        #spreadsheetSection { 
            display: none; 
            margin-top: 40px;
            flex-grow: 1;
            overflow: auto;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }
        th { 
            background-color: #f2f2f2; 
            text-align: center;
        }
        input[type="text"] { 
            width: 100%; 
            box-sizing: border-box; 
        }
        .button-container { 
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f8f8f8;
            padding: 10px;
            text-align: center;
        }
        .unit-input { 
            width: 100%; 
            box-sizing: border-box;
            text-align: center;
        }
        .unit-row th {
            padding: 4px;
            background-color: #e6e6e6;
        }
        .spacer {
            height: 60px; /* Adjust based on the height of your button container */
        }
    </style>
</head>
<body>
    <h1>CSV Mapping and Editor with Units</h1>
    
    <!-- Mapping Section -->
    <div id="mappingSection">
        <div id="csvInfo"></div>
        <div id="mappingContainer"></div>
        <button onclick="confirmMapping()" id="confirmButton" style="display: none;">Confirm Mapping</button>
    </div>

    <!-- Spreadsheet Section -->
    <div id="spreadsheetSection">
        <h2>Data Editor</h2>
        <div id="tableContainer"></div>
    </div>
    
    <div class="spacer"></div>

    <div class="button-container">
        <button onclick="addRow()">Add Row</button>
        <button onclick="removeRow()">Remove Row</button>
        <button onclick="copyToClipboard()">Copy to Clipboard</button>
        <button onclick="pasteFromClipboard()">Paste from Clipboard</button>
        <button onclick="saveData()">Save</button>
        <button onclick="clearData()">Clear</button>
    </div>

    <script>
        // Initialize expectedHeaders and defaultUnits with data passed from the server
        const expectedHeaders = JSON.parse('{{ expectedHeaders }}');
        const defaultUnits = JSON.parse('{{ defaultUnits }}');
        let currentHeaders = ['None'];
        let csvData = [];
        let mappedData = [];
        let currentUnits = [...defaultUnits];

        // CSV Mapping Functions
        function parseCSV(text) {
            const lines = text.split('\\n');
            return lines.map(line => line.split(',').map(value => value.trim()));
        }

        function columnsMatch() {
            return expectedHeaders.every(header => currentHeaders.includes(header));
        }

        function loadCSV() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    csvData = parseCSV(e.target.result);
                    currentHeaders = ['None'].concat(csvData[0]);
                    displayCSVInfo();
                    if (columnsMatch()) {
                        // All expected headers are present; skip mapping
                        confirmMapping();
                    } else {
                        populateMapping();
                        document.getElementById('confirmButton').style.display = 'inline-block';
                    }
                };
                reader.readAsText(file);
            } else {
                alert('Please select a CSV file first.');
            }
        }

        function displayCSVInfo() {
            const infoDiv = document.getElementById('csvInfo');
            infoDiv.innerHTML = ``;
                //<p>CSV loaded successfully:</p>
                //<p>Number of columns: ${currentHeaders.length - 1}</p>
                //<p>Number of rows: ${csvData.length}</p>
                //<p>Headers: ${currentHeaders.slice(1).join(', ')}</p>
            //`;
        }

        function populateMapping() {
            const container = document.getElementById('mappingContainer');
            container.innerHTML = ''; // Clear previous mapping
            expectedHeaders.forEach(header => {
                const row = document.createElement('div');
                row.className = 'mapping-row';
                row.innerHTML = `
                    <label>${header}:</label>
                    <select id="${header}-select">
                        ${currentHeaders.map(ch => `<option value="${ch}">${ch}</option>`).join('')}
                    </select>
                `;
                container.appendChild(row);
            });
        }

        function confirmMapping(autoMap = false) {
            const mapping = {};

            // Automatically map headers if autoMap is true
            if (autoMap) {
                expectedHeaders.forEach(header => {
                    mapping[header] = header; // Direct mapping
                });
            } else {
                // Manual mapping through the UI
                expectedHeaders.forEach(header => {
                    const select = document.getElementById(`${header}-select`);
                    if (select) {
                        mapping[header] = select.value;
                    }
                });
            }
            console.log('Mapping:', mapping);

            // Apply mapping to data
            if (csvData.length > 0) {
                mappedData = csvData.slice(1).map(row => {
                    return expectedHeaders.map(header => {
                        const index = csvData[0].indexOf(mapping[header]);
                        return index !== -1 ? row[index] : '';
                    });
                });
            } else if (initialCSVData.length > 0) {
                mappedData = initialCSVData.slice(1);
            } else {
                // If no data is available, create an empty row
                mappedData = [new Array(expectedHeaders.length).fill('')];
            }

            // Show spreadsheet section and render table
            document.getElementById('spreadsheetSection').style.display = 'block';
            renderTable();
        }

        function renderTable() {
            const container = document.getElementById('tableContainer');
            let html = '<table>';
            
            // Units row
            html += '<tr class="unit-row">';
            expectedHeaders.forEach((header, index) => {
                html += `<th>
                    <input type="text" class="unit-input" id="unit-${index}" 
                           value="${currentUnits[index]}" 
                           ${defaultUnits[index] === "" ? 'disabled' : ''}
                           oninput="updateUnit(${index}, this.value)">
                </th>`;
            });
            html += '</tr>';

            // Header row
            html += '<tr>';
            expectedHeaders.forEach(header => {
                html += `<th>${header}</th>`;
            });
            html += '</tr>';

            // Data rows
            mappedData.forEach((row, rowIndex) => {
                html += '<tr>';
                expectedHeaders.forEach((header, colIndex) => {
                    html += `<td><input type="text" value="${row[colIndex] || ''}" oninput="updateData(${rowIndex}, ${colIndex}, this.value)"></td>`;
                });
                html += '</tr>';
            });
            html += '</table>';
            container.innerHTML = html;
        }

        function updateUnit(index, value) {
            currentUnits[index] = value;
        }

        function updateData(row, col, value) {
            mappedData[row][col] = value;
        }

        function addRow() {
            mappedData.push(new Array(expectedHeaders.length).fill(''));
            renderTable();
        }

        function removeRow() {
            if (mappedData.length > 0) {
                mappedData.pop();
                renderTable();
            }
        }

        function copyToClipboard() {
            const text = [expectedHeaders.join('\\t')].concat(mappedData.map(row => row.join('\\t'))).join('\\n');
            navigator.clipboard.writeText(text).then(() => {
                alert('Data copied to clipboard');
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        }

        function pasteFromClipboard() {
            navigator.clipboard.readText().then(text => {
                const rows = text.split('\\n');
                mappedData = rows.slice(1).map(row => row.split('\\t'));
                renderTable();
            }).catch(err => {
                console.error('Failed to paste: ', err);
            });
        }

        function saveData() {
            // Append units to data
            const dataWithUnits = mappedData.map(row => 
                row.map((value, index) => 
                    currentUnits[index] ? `${value} ${currentUnits[index]}` : value
                )
            );

            // Send data to server
            fetch('/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ data: dataWithUnits })
            }).then(response => {
                if (response.ok) {
                    alert('Data saved successfully');
                } else {
                    alert('Failed to save data');
                }
            }).catch(error => console.error('Error:', error));
        }

        function clearData() {
            // Send clear instruction to server
            fetch('/clear', {
                method: 'POST'
            }).then(response => {
                if (response.ok) {
                    alert('Data cleared successfully');
                } else {
                    alert('Failed to clear data');
                }
            }).catch(error => console.error('Error:', error));
        }
    </script>
</body>
<script>
    // Load initial CSV data
    const initialCSVData = JSON.parse('{{ initialCSVData }}');
    window.addEventListener('load', function() {
        if (initialCSVData.length > 0) {
            csvData = initialCSVData;
            currentHeaders = ['None'].concat(csvData[0]);
            displayCSVInfo();
            if (columnsMatch()) {
                // Automatically map headers and proceed to the spreadsheet
                confirmMapping(true);
            } else {
                populateMapping();
                document.getElementById('confirmButton').style.display = 'inline-block';
            }
        }
    });

</script>
</html>
'''

# Example usage:
# custom_edit(app, ["MD", "INC", "AZM"], ["m", "deg", "deg"], "path/to/your/file.csv")
