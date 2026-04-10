// 🌟 穩定版 V2 引擎
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0';
import stringSimilarity from 'https://esm.sh/string-similarity@4.0.2';
// 🌟 匯入繁簡轉換神器 OpenCC
import * as OpenCC from 'https://esm.sh/opencc-js@1.0.5';

// 配置環境
env.allowLocalModels = false;

// 建立轉換器 (cn: 簡體 -> tw: 台灣繁體)
const converter = OpenCC.Converter({ from: 'cn', to: 'tw' });

// 綁定 UI 元素
const csvFileInput = document.getElementById('csvFile');
const folderFileInput = document.getElementById('folderFile'); 
const startBtn = document.getElementById('startBtn');
const statusHeader = document.getElementById('statusHeader');
const statusLog = document.getElementById('statusLog');
const progressBar = document.getElementById('progressBar');
const progressWrapper = document.getElementById('progressWrapper');
const resultSection = document.getElementById('resultSection');

// 綁定新功能的 UI 元素
const modeA = document.getElementById('modeA');
const modeB = document.getElementById('modeB');
const csvUploadBlock = document.getElementById('csvUploadBlock');
const modelSelect = document.getElementById('modelSelect');

// 互動小巧思：選擇 Mode A 時，隱藏 CSV 上傳區塊
modeA.addEventListener('change', () => { csvUploadBlock.style.display = 'none'; });
modeB.addEventListener('change', () => { csvUploadBlock.style.display = 'block'; });

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
    const isModeB = modeB.checked; // 判斷是否為「比對模式」
    const selectedModel = modelSelect.value; // 取得使用者選擇的模型

    // 防呆檢查
    if (isModeB && !csvFileInput.files[0]) {
        alert("模式 B 需要上傳文本 CSV 供比對！");
        return;
    }
    if (folderFileInput.files.length === 0) {
        alert("請選擇包含音檔的語音資料夾！");
        return;
    }

    startBtn.disabled = true;
    resultSection.innerHTML = '';
    statusLog.innerHTML = '';
    let textList = [];
    let matchResults = [];

    try {
        // Step 1: 如果是模式 B，才需要解析 CSV
        if (isModeB) {
            statusHeader.textContent = "📄 正在解析文本資料...";
            const csvContent = await csvFileInput.files[0].text();
            Papa.parse(csvContent, {
                skipEmptyLines: true,
                complete: (results) => {
                    textList = results.data.map(row => row[0].trim()).filter(t => t.length > 0);
                }
            });
            log(`讀取到 ${textList.length} 句文本供比對。`);
        }

        // Step 2: 根據選單載入指定的 AI 模型
        statusHeader.textContent = `🤖 正在載入 AI 模型 (${selectedModel.split('-')[1]})...`;
        const transcriber = await pipeline('automatic-speech-recognition', selectedModel, {
            progress_callback: (info) => {
                if (info.status === 'progress') {
                    updateProgress(info.progress);
                    statusHeader.textContent = `⏳ 下載模型資源中... ${Math.round(info.progress)}%`;
                }
                if (info.status === 'ready') {
                    statusHeader.textContent = "✅ 模型已就緒，準備開始辨識。";
                    updateProgress(100);
                }
            }
        });

        // Step 3: 讀取本地資料夾
        statusHeader.textContent = "📁 讀取資料夾音檔...";
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        const allFiles = Array.from(folderFileInput.files);
        const audioFiles = allFiles.filter(file => 
            file.name.toLowerCase().endsWith('.wav') || file.name.toLowerCase().endsWith('.mp3')
        );

        if (audioFiles.length === 0) throw new Error("資料夾內沒有找到有效的音檔。");

        // Step 4: 循環處理
        for (let i = 0; i < audioFiles.length; i++) {
            const file = audioFiles[i]; 
            const fileName = file.name;
            statusHeader.textContent = `🎧 正在辨識 (${i+1}/${audioFiles.length})`;
            log(`處理中: ${fileName}`);

            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const float32Data = audioBuffer.getChannelData(0);

            // AI 語音轉文字
            const whisperRes = await transcriber(float32Data, {
                language: 'chinese',
                task: 'transcribe'
            });
            
            // 強制轉換為台灣繁體
            let heardText = whisperRes.text.trim();
            heardText = converter(heardText);

            // 根據模式決定輸出的資料格視
            if (isModeB) {
                // 模式 B：進行比對
                const match = stringSimilarity.findBestMatch(heardText, textList);
                const score = Math.round(match.bestMatch.rating * 100);
                matchResults.push({
                    "匹配文本": match.bestMatch.target,
                    "語音檔名": fileName,
                    "辨識內容": heardText,
                    "置信度": `${score}%`
                });
            } else {
                // 模式 A：純轉譯，不比對
                matchResults.push({
                    "語音檔名": fileName,
                    "辨識內容 (純繁體)": heardText
                });
            }
            
            updateProgress(((i + 1) / audioFiles.length) * 100);
        }

        // Step 5: 下載結果
        statusHeader.textContent = "🎉 任務完成！";
        const finalCsv = Papa.unparse(matchResults);
        const blob = new Blob(["\ufeff" + finalCsv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        resultSection.innerHTML = `<a href="${url}" download="Result.csv" class="btn-success">⬇️ 下載輸出結果 CSV</a>`;

    } catch (err) {
        statusHeader.textContent = "❌ 發生錯誤";
        log(err.message);
    } finally {
        startBtn.disabled = false;
    }
});
