# Performance Database

Complete registry of stress test results.

<div style="display: flex; gap: 10px; margin-bottom: 20px;">
  <select id="gpuFilter" onchange="applyFilters()" style="padding: 10px; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px;">
    <option value="all">All GPUs</option>
    </select>

  <select id="testFilter" onchange="applyFilters()" style="padding: 10px; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px;">
    <option value="all">All Tests</option>
    </select>

  <input type="text" id="textSearch" onkeyup="applyFilters()" placeholder="Search..." style="padding: 10px; flex-grow: 1; background: #222; color: #fff; border: 1px solid #444; border-radius: 4px;">
</div>

<div style="overflow-x:auto;">
  <table id="benchmarkTable">
    <thead>
      <tr style="cursor: pointer;">
        <th onclick="sortTable(0)">GPU Model &#8597;</th>
        <th onclick="sortTable(1)">Test Name &#8597;</th>
        <th onclick="sortTable(2)">Ver &#8597;</th>
        <th onclick="sortTable(3)">Peak Temp &#8597;</th>
        <th onclick="sortTable(4)">Peak Power &#8597;</th>
        <th onclick="sortTable(5)">Avg Clock &#8597;</th>
        <th onclick="sortTable(6)">Date &#8597;</th>
      </tr>
    </thead>
    <tbody>
      </tbody>
  </table>
</div>

<script src="../js/tables.js"></script>
