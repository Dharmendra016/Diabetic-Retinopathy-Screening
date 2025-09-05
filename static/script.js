document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const diagnoseButton = document.getElementById('diagnoseButton');
    const loadingMessage = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const resultsControls = document.getElementById('results-controls');
    const downloadCSVButton = document.getElementById('downloadCSV');
    
    let allResults = [];

    diagnoseButton.addEventListener('click', async () => {
        const files = imageInput.files;
        if (files.length === 0) {
            alert('Please select one or more images first.');
            return;
        }

        loadingMessage.classList.remove('hidden');
        resultsDiv.innerHTML = '';
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        resultsControls.classList.add('hidden');

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

        try {
            const response = await fetch('/api/diagnose', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error.');
            }

            allResults = await response.json();
            displayResults(allResults);
            resultsDiv.classList.remove('hidden');
            resultsControls.classList.remove('hidden');
            
        } catch (error) {
            console.error('Diagnosis failed:', error);
            errorDiv.classList.remove('hidden');
        } finally {
            loadingMessage.classList.add('hidden');
        }
    });

    downloadCSVButton.addEventListener('click', () => {
        if (allResults.length === 0) return;
        
        const headers = ["Filename", "Diagnosis", "Confidence"];
        const rows = allResults.map(res => [
            res.filename,
            res.prediction,
            res.confidence
        ]);

        let csvContent = "data:text/csv;charset=utf-8," 
        + headers.join(",") + "\n"
        + rows.map(e => e.join(",")).join("\n");
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "dr_referral_report.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    function getTriageClass(prediction) {
        if (prediction === 'Normal') return 'no-dr';
        if (prediction === 'Mild' || prediction === 'Moderate') return 'mild-moderate';
        return 'severe-proliferative';
    }

    function displayResults(results) {
        resultsDiv.innerHTML = '';
        results.forEach(result => {
            const triageClass = getTriageClass(result.prediction);
            const card = document.createElement('div');
            card.classList.add('result-card', triageClass);
            card.innerHTML = `
                <div class="result-text">
                    <h4>${result.filename}</h4>
                    <p><strong>Diagnosis:</strong> ${result.prediction}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                </div>
                <div class="heatmap-container">
                    <p>Lesion Heatmap</p>
                    <img src="data:image/jpeg;base64,${result.heatmap}" alt="Heatmap">
                </div>
            `;
            resultsDiv.appendChild(card);
        });
    }
});