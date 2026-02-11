# Performance Database

Complete registry of stress test results.

<div style="display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;">
  
  <select id="gpuFilter" onchange="applyFilters()" style="padding: 10px; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px; flex: 1;">
    <option value="all">All GPUs</option>
  </select>

  <select id="testFilter" onchange="applyFilters()" style="padding: 10px; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px; flex: 1;">
    <option value="all">All Tests</option>
  </select>

  <div style="position: relative; display: inline-block;">
    <button onclick="toggleColumnMenu()" style="padding: 10px; background: #333; color: #fff; border: 1px solid #555; border-radius: 4px; cursor: pointer;">
      Columns &#9662;
    </button>
    <div id="columnMenu" style="display: none; position: absolute; background-color: #1a1a1a; min-width: 200px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.5); z-index: 1; padding: 10px; border: 1px solid #444; border-radius: 4px;">
      </div>
  </div>

  <input type="text" id="textSearch" onkeyup="applyFilters()" placeholder="Search..." style="padding: 10px; flex-grow: 2; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px;">

</div>

<div style="overflow-x:auto;">
  <table id="benchmarkTable">
    <thead>
      </thead>
    <tbody>
      </tbody>
  </table>
</div>

<script src="../js/tables.js"></script>
