let rawData = []; // Store full history
let bestRuns = []; // Store only best scores

document.addEventListener("DOMContentLoaded", function () {
    const dataUrl = "../assets/web_data.json";

    fetch(dataUrl)
        .then(response => response.json())
        .then(data => {
            rawData = data;
            // 1. Process data to find best runs
            bestRuns = getBestRunsOnly(rawData);
            
            // 2. Populate filters based on the best runs
            populateFilters(bestRuns);
            
            // 3. Render the best runs by default
            renderTable(bestRuns);
        })
        .catch(err => console.error("Error loading benchmark data:", err));
});

// --- NEW: Aggregation Logic ---
function getBestRunsOnly(data) {
    const groups = {};

    data.forEach(row => {
        // Create a unique key for this GPU + Test combo
        const key = `${row.gpu}|${row.test}`;

        // If we haven't seen this combo, or if this run has a higher score, save it
        if (!groups[key] || parseFloat(row.score) > parseFloat(groups[key].score)) {
            groups[key] = row;
        }
    });

    // Convert the object back to an array
    return Object.values(groups);
}

// --- Populate Dropdown Menus ---
function populateFilters(data) {
    const gpuSet = new Set();
    const testSet = new Set();

    data.forEach(row => {
        gpuSet.add(row.gpu);
        testSet.add(row.test);
    });

    const gpuSelect = document.getElementById("gpuFilter");
    const testSelect = document.getElementById("testFilter");

    // Clear existing options (in case of reload)
    gpuSelect.innerHTML = '<option value="all">All GPUs</option>';
    testSelect.innerHTML = '<option value="all">All Tests</option>';

    // Add GPUs
    Array.from(gpuSet).sort().forEach(gpu => {
        const option = document.createElement("option");
        option.value = gpu;
        option.textContent = gpu;
        gpuSelect.appendChild(option);
    });

    // Add Tests
    Array.from(testSet).sort().forEach(test => {
        const option = document.createElement("option");
        option.value = test;
        option.textContent = test;
        testSelect.appendChild(option);
    });
}

// --- Render Table ---
function renderTable(data) {
    const tableBody = document.querySelector("#benchmarkTable tbody");
    tableBody.innerHTML = ""; 

    if (data.length === 0) {
        tableBody.innerHTML = "<tr><td colspan='7' style='text-align:center; padding: 20px;'>No results found</td></tr>";
        return;
    }

    data.forEach(row => {
        const tr = document.createElement("tr");
        
        let scoreDisplay = (row.score === "N/A" || row.score === undefined) ? "N/A" : `${row.score} ${row.unit}`;
        
        // Color coding
        const tempColor = getColorForTemp(row.temp_max);

        tr.innerHTML = `
            <td style="font-weight:bold; color:#fff;">${row.gpu}</td>
            <td style="color:#00f3ff;">${row.test}</td>
	    <td style="font-size:0.8em; color:#aaa;">${row.version || "Legacy"}</td> 
	    <td style="font-weight:bold;">${scoreDisplay}</td>
            <td style="color:${tempColor};">${row.temp_max ? row.temp_max + "°C" : "N/A"}</td>
            <td>${row.power_max ? row.power_max + " W" : "N/A"}</td>
            <td>${row.clock_avg ? row.clock_avg + " MHz" : "N/A"}</td>
            <td style="font-size:0.85em; color:#888;">${row.date || ""}</td>
        `;
        tableBody.appendChild(tr);
    });
}

// --- Filter Logic ---
function applyFilters() {
    const gpuVal = document.getElementById("gpuFilter").value;
    const testVal = document.getElementById("testFilter").value;
    const searchVal = document.getElementById("textSearch").value.toLowerCase();

    // Always filter against 'bestRuns' so we don't show duplicates
    const filtered = bestRuns.filter(row => {
        const gpuMatch = (gpuVal === "all") || (row.gpu === gpuVal);
        const testMatch = (testVal === "all") || (row.test === testVal);
        const rowString = `${row.gpu} ${row.test} ${row.date}`.toLowerCase();
        const searchMatch = rowString.includes(searchVal);

        return gpuMatch && testMatch && searchMatch;
    });

    renderTable(filtered);
}

// --- Helpers ---
function getColorForTemp(temp) {
    if (!temp) return "#fff";
    if (temp < 60) return "#00e676"; // Green
    if (temp < 80) return "#ffeb3b"; // Yellow
    if (temp >= 85) return "#ff3d00"; // Red
    return "#fff";
}

function sortTable(n) {
    const table = document.getElementById("benchmarkTable");
    let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
    switching = true;
    dir = "asc"; 
    
    while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
            shouldSwitch = false;
            x = rows[i].getElementsByTagName("TD")[n];
            y = rows[i + 1].getElementsByTagName("TD")[n];
            
            let xVal = x.textContent.toLowerCase().replace(/[^0-9a-z\.\-]/g, ""); 
            let yVal = y.textContent.toLowerCase().replace(/[^0-9a-z\.\-]/g, "");

            let xNum = parseFloat(xVal);
            let yNum = parseFloat(yVal);
            let isNumeric = !isNaN(xNum) && !isNaN(yNum);

            if (isNumeric) {
                 if (dir == "asc") {
                    if (xNum > yNum) { shouldSwitch = true; break; }
                } else if (dir == "desc") {
                    if (xNum < yNum) { shouldSwitch = true; break; }
                }
            } else {
                xVal = x.textContent.toLowerCase();
                yVal = y.textContent.toLowerCase();
                if (dir == "asc") {
                    if (xVal > yVal) { shouldSwitch = true; break; }
                } else if (dir == "desc") {
                    if (xVal < yVal) { shouldSwitch = true; break; }
                }
            }
        }
        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
            switchcount ++; 
        } else {
            if (switchcount == 0 && dir == "asc") {
                dir = "desc";
                switching = true;
            }
        }
    }
}
