// --- CONFIGURATION: Define all available columns ---
const COL_DEFS = [
    { key: "gpu",        label: "GPU Model",   visible: true },
    { key: "test",       label: "Test Name",   visible: true },
    { key: "version",    label: "Ver",         visible: true },
    { key: "score",      label: "Score",       visible: true },
    { key: "temp_max",   label: "Peak Temp",   visible: true },
    { key: "power_max",  label: "Peak Power",  visible: true },
    { key: "clock_avg",  label: "Avg Clock",   visible: true },
    { key: "date",       label: "Date",        visible: true },
    // --- Hidden by default (Pro Metrics) ---
    { key: "efficiency", label: "Efficiency (MB/J)", visible: false },
    { key: "temp_mem",   label: "Mem Temp",    visible: false },
    { key: "fan_max",    label: "Fan %",       visible: false },
    { key: "pcie_gen",   label: "PCIe Gen",    visible: false },
    { key: "pcie_width", label: "PCIe Width",  visible: false },
    { key: "throttle",   label: "Limit Reason",visible: false },
    { key: "volts_core", label: "Core (mV)",   visible: false },
    { key: "volts_soc",  label: "SoC (mV)",    visible: false },
];

let rawData = [];
let bestRuns = [];
let currentSort = { key: 'score', dir: 'desc' }; // Default sort

document.addEventListener("DOMContentLoaded", function () {
    const dataUrl = "../assets/web_data.json";

    fetch(dataUrl)
        .then(response => response.json())
        .then(data => {
            rawData = data;
            bestRuns = getBestRunsOnly(rawData);
            
            initColumnMenu();     // Setup the checkboxes
            populateFilters(bestRuns);
            renderTable(bestRuns);
        })
        .catch(err => console.error("Error loading benchmark data:", err));
});

// --- 1. Aggregation Logic ---
function getBestRunsOnly(data) {
    const groups = {};
    data.forEach(row => {
        const key = `${row.gpu}|${row.test}`;
        if (!groups[key] || parseFloat(row.score) > parseFloat(groups[key].score)) {
            groups[key] = row;
        }
    });
    return Object.values(groups);
}

// --- 2. Dynamic Table Rendering ---
function renderTable(data) {
    const table = document.getElementById("benchmarkTable");
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    // A. Build Header based on Visible Columns
    thead.innerHTML = "";
    let headerRow = document.createElement("tr");
    headerRow.style.cursor = "pointer";

    COL_DEFS.forEach(col => {
        if (col.visible) {
            let th = document.createElement("th");
            th.innerHTML = `${col.label} &#8597;`;
            th.onclick = () => sortData(col.key);
            headerRow.appendChild(th);
        }
    });
    thead.appendChild(headerRow);

    // B. Build Rows
    tbody.innerHTML = "";
    if (data.length === 0) {
        let visibleCount = COL_DEFS.filter(c => c.visible).length;
        tbody.innerHTML = `<tr><td colspan='${visibleCount}' style='text-align:center; padding: 20px;'>No results found</td></tr>`;
        return;
    }

    data.forEach(row => {
        const tr = document.createElement("tr");
        
        COL_DEFS.forEach(col => {
            if (col.visible) {
                let td = document.createElement("td");
                let val = row[col.key];

                // --- Special Formatting ---
                if (col.key === "score") {
                    val = (val === "N/A" || val === undefined) ? "N/A" : `${val} ${row.unit}`;
                    td.style.fontWeight = "bold";
                } 
                else if (col.key === "gpu") {
                    td.style.fontWeight = "bold";
                    td.style.color = "#fff";
                }
                else if (col.key === "test") {
                    td.style.color = "#00f3ff";
                }
                else if (col.key.includes("temp")) {
                    td.style.color = getColorForTemp(val);
                    if(val) val += "°C";
                }
                else if (col.key.includes("power")) {
                    if(val) val += " W";
                }
                else if (col.key === "version") {
                    td.style.fontSize = "0.8em";
                    td.style.color = "#aaa";
                    if(!val) val = "Legacy";
                }
                
                if (val === undefined || val === 0 || val === "0") val = "N/A";
                td.textContent = val;
                tr.appendChild(td);
            }
        });
        tbody.appendChild(tr);
    });
}

// --- 3. Sorting Logic ---
function sortData(key) {
    // Toggle direction if clicking same header
    if (currentSort.key === key) {
        currentSort.dir = currentSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.key = key;
        currentSort.dir = 'desc'; // Default to high-to-low for new columns
    }

    // Sort the global filtered list (so sorting works after filtering)
    // We re-run applyFilters which will trigger the sort
    applyFilters();
}

// --- 4. Filtering Logic ---
function applyFilters() {
    const gpuVal = document.getElementById("gpuFilter").value;
    const testVal = document.getElementById("testFilter").value;
    const searchVal = document.getElementById("textSearch").value.toLowerCase();

    let filtered = bestRuns.filter(row => {
        const gpuMatch = (gpuVal === "all") || (row.gpu === gpuVal);
        const testMatch = (testVal === "all") || (row.test === testVal);
        const rowString = Object.values(row).join(" ").toLowerCase();
        const searchMatch = rowString.includes(searchVal);
        return gpuMatch && testMatch && searchMatch;
    });

    // Apply Sorting
    filtered.sort((a, b) => {
        let valA = a[currentSort.key];
        let valB = b[currentSort.key];

        // Handle N/A or missing
        if (valA === undefined || valA === "N/A") valA = -999999;
        if (valB === undefined || valB === "N/A") valB = -999999;

        // Numeric parsing
        let numA = parseFloat(valA);
        let numB = parseFloat(valB);

        if (!isNaN(numA) && !isNaN(numB)) {
            valA = numA;
            valB = numB;
        } else {
            valA = String(valA).toLowerCase();
            valB = String(valB).toLowerCase();
        }

        if (valA < valB) return currentSort.dir === 'asc' ? -1 : 1;
        if (valA > valB) return currentSort.dir === 'asc' ? 1 : -1;
        return 0;
    });

    renderTable(filtered);
}

// --- 5. Column Visibility UI ---
function initColumnMenu() {
    const menu = document.getElementById("columnMenu");
    menu.innerHTML = "";

    COL_DEFS.forEach((col, index) => {
        let div = document.createElement("div");
        div.style.marginBottom = "5px";
        
        let label = document.createElement("label");
        label.style.cursor = "pointer";
        label.style.display = "flex";
        label.style.alignItems = "center";
        
        let check = document.createElement("input");
        check.type = "checkbox";
        check.checked = col.visible;
        check.style.marginRight = "10px";
        
        check.onchange = () => {
            COL_DEFS[index].visible = check.checked;
            applyFilters(); // Re-render table
        };

        label.appendChild(check);
        label.appendChild(document.createTextNode(col.label));
        div.appendChild(label);
        menu.appendChild(div);
    });
}

function toggleColumnMenu() {
    const menu = document.getElementById("columnMenu");
    menu.style.display = menu.style.display === "block" ? "none" : "block";
}

// Close menu if clicked outside
window.onclick = function(event) {
    if (!event.target.matches('button')) {
        const menu = document.getElementById("columnMenu");
        if (menu && menu.style.display === "block" && !menu.contains(event.target)) {
            menu.style.display = "none";
        }
    }
}

// --- Helpers ---
function populateFilters(data) {
    const gpuSet = new Set();
    const testSet = new Set();
    data.forEach(row => {
        gpuSet.add(row.gpu);
        testSet.add(row.test);
    });

    const gpuSelect = document.getElementById("gpuFilter");
    const testSelect = document.getElementById("testFilter");

    gpuSelect.innerHTML = '<option value="all">All GPUs</option>';
    testSelect.innerHTML = '<option value="all">All Tests</option>';

    Array.from(gpuSet).sort().forEach(gpu => {
        let op = document.createElement("option"); op.value = gpu; op.textContent = gpu;
        gpuSelect.appendChild(op);
    });
    Array.from(testSet).sort().forEach(test => {
        let op = document.createElement("option"); op.value = test; op.textContent = test;
        testSelect.appendChild(op);
    });
}

function getColorForTemp(temp) {
    if (!temp) return "#fff";
    if (temp < 60) return "#00e676"; // Green
    if (temp < 80) return "#ffeb3b"; // Yellow
    if (temp >= 85) return "#ff3d00"; // Red
    return "#fff";
}
