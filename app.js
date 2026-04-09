import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0';
// ?? 就是新增下面這一行，讓程式直接把比對工具抓進來！
import stringSimilarity from 'https://esm.sh/string-similarity@4.0.2';

// 配置環境
env.allowLocalModels = false;

// 綁定 UI 元素
const csvFileInput = document.getElementById('csvFile');
const zipFileInput = document.getElementById('zipFile');
const startBtn = document.getElementById('startBtn');
const statusHeader = document.getElementById('statusHeader');
const statusLog = document.getElementById('statusLog');
const progressBar = document.getElementById('progressBar');
const progressWrapper = document.getElementById('progressWrapper');
const resultSection = document.getElementById('resultSection');

function log(msg) {
    const div = document.createElement('div');
    div.textContent = `> ${msg}`;
    statusLog.prepend(div);
}

function updateProgress(percent) {
    progressWrapper.style.display = 'block';
    progressBar.style.width = `${percent}%`;
}

startBtn.addEventListener('click', async () => {
    if (!csvFileInput.files[0] || !zipFileInput.files[0]) {
        alert("請確認已上傳 CSV 與 ZIP 檔案！");
        return;
    }

    startBtn.disabled = true;
    resultSection.innerHTML = '';
    statusLog.innerHTML = '';
    let textList = [];
    let matchResults = [];

    try {
        // Step 1: 解析 CSV
        statusHeader.textContent = "?? 正在解析文本資料...";
        const csvContent = await csvFileInput.files[0].text();
        Papa.parse(csvContent, {
            skipEmptyLines: true,
            complete: (results) => {
                textList = results.data.map(row => row[0].trim()).filter(t => t.length > 0);
            }
        });
        log(`讀取到 ${textList.length} 句文本。`);

        // Step 2: 載入 AI 模型
        statusHeader.textContent = "?? 正在載入 AI 模型 (Whisper-base)...";
        const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
            progress_callback: (info) => {
                if (info.status === 'downloading') {
                    updateProgress(info.progress);
                    statusHeader.textContent = `? 下載模型資源中... ${Math.round(info.progress)}%`;
                }
                if (info.status === 'ready') {
                    statusHeader.textContent = "? 模型已就緒，準備開始辨識。";
                    updateProgress(100);
                }
            }
        });

        // Step 3: 解壓縮音檔
        statusHeader.textContent = "?? 解壓縮語音包...";
        const zip = new JSZip();
        const folder = await zip.loadAsync(zipFileInput.files[0]);
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        const audioFiles = Object.keys(folder.files).filter(name => 
            name.toLowerCase().endsWith('.wav') || name.toLowerCase().endsWith('.mp3')
        );

        if (audioFiles.length === 0) throw new Error("ZIP 內無效音檔。");

        // Step 4: 循環處理
        for (let i = 0; i < audioFiles.length; i++) {
            const fileName = audioFiles[i];
            statusHeader.textContent = `?? 正在辨識 (${i+1}/${audioFiles.length})`;
            log(`處理中: ${fileName}`);

            const arrayBuffer = await folder.files[fileName].async("arraybuffer");
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const float32Data = audioBuffer.getChannelData(0);

            // AI 語音轉文字
            const whisperRes = await transcriber(float32Data, {
                language: 'chinese',
                task: 'transcribe'
            });
            const heardText = whisperRes.text.trim();

            // 文本相似度比對
            const match = stringSimilarity.findBestMatch(heardText, textList);
            const score = Math.round(match.bestMatch.rating * 100);

            matchResults.push({
                "匹配文本": match.bestMatch.target,
                "語音檔名": fileName,
                "辨識內容": heardText,
                "置信度": `${score}%`
            });
            
            updateProgress(((i + 1) / audioFiles.length) * 100);
        }

        // Step 5: 下載結果
        statusHeader.textContent = "? 任務完成！";
        const finalCsv = Papa.unparse(matchResults);
        const blob = new Blob(["\ufeff" + finalCsv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        resultSection.innerHTML = `<a href="${url}" download="Matching_Result.csv" class="btn-success">?? 下載配對結果 CSV</a>`;

    } catch (err) {
        statusHeader.textContent = "? 發生錯誤";
        log(err.message);
    } finally {
        startBtn.disabled = false;
    }
});