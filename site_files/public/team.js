
console.log("teams.js loaded");
function alphabetizeSelect(selectId) {
    const select = document.getElementById(selectId);
    if (!select) return;

    const options = Array.from(select.options);

    // Keep the first option (placeholder)
    const firstOption = options.shift();

    options.sort((a, b) =>
        a.text.localeCompare(b.text, undefined, { sensitivity: 'base' })
    );

    select.innerHTML = "";
    select.appendChild(firstOption);
    options.forEach(option => select.appendChild(option));
}

document.addEventListener("DOMContentLoaded", function () {
    // Alphabetize both dropdowns
    alphabetizeSelect("team1");
    alphabetizeSelect("team2");

    // Make them searchable with Select2
    $("#team1, #team2").select2({
        placeholder: "Select a team",
        allowClear: true,
        width: "100%"
    });
});