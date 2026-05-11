// WebGPU 專用版 app.js
// 重點：只使用 WebGPU，不 fallback 到 WASM。

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import stringSimilarity from 'https://esm.sh/string-similarity@4.0.2';
import * as OpenCC from 'https://esm.sh/opencc-js@1.0.5';

env.allowLocalModels = false;

const converter = OpenCC.Converter({ from: 'cn', to: 'tw' });

const csvFileInput = document.getElementById('csvFile');
const folderFileInput = document.getElementById('folderFile');
const startBtn = document.getElementById('startBtn');
const statusHeader = document.getElementById('statusHeader');
const statusLog = document.getElementById('statusLog');
const progressBar = document.getElementById('progressBar');
const progressWrapper = document.getElementById('progressWrapper');
const resultSection = document.getElementById('resultSection');

const modeA = document.getElementById('modeA');
const modeB = document.getElementById('modeB');
const csvUploadBlock = document.getElementById('csvUploadBlock');
const modelSelect = document.getElementById('modelSelect');

const MODEL_MAP = {
    'Xenova/whisper-tiny': 'onnx-community/whisper-tiny',
    'Xenova/whisper-base': 'onnx-community/whisper-base',
    'Xenova/whisper-small': 'onnx-community/whisper-small',
    'Xenova/whisper-large-v2': 'onnx-community/whisper-large-v2'
};

let cachedTranscriber = null;
let cachedModelName = null;

modeA.addEventListener('change', () => {
    csvUploadBlock.style.display = 'none';
});

modeB.addEventListener('change', () => {
    csvUploadBlock.style.display = 'block';
});

function log(msg) {
    const div = document.createElement('div');
    div.textContent = `> ${msg}`;
    statusLog.prepend(div);
}

function updateProgress(percent) {
    const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
    progressWrapper.style.display = 'block';
    progressBar.style.width = `${safePercent}%`;
}

function setStatus(text) {
    statusHeader.textContent = text;
}

function resolveModelName(selectedModel) {
    return MODEL_MAP[selectedModel] || selectedModel;
}

async function ensureWebGPU() {
    if (!window.isSecureContext) {
        throw new Error('WebGPU 需要 HTTPS 安全環境。請使用 GitHub Pages / Cloudflare Pages / https 網址。');
    }

    if (!navigator.gpu) {
        throw new Error('此瀏覽器沒有 WebGPU API。請使用新版 Chrome / Edge，或支援 WebGPU 的瀏覽器。');
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
    });

    if (!adapter) {
        throw new Error('找不到可用的 WebGPU GPU Adapter。請檢查顯示卡/內顯、驅動程式、瀏覽器硬體加速。');
    }

    console.log('WebGPU adapter:', adapter);
    return adapter;
}

async function createWebGPUTranscriber(modelName) {
    await ensureWebGPU();

    if (cachedTranscriber && cachedModelName === modelName) {
        log(`沿用已載入模型：${modelName}`);
        return cachedTranscriber;
    }

    cachedTranscriber = null;
    cachedModelName = null;

    setStatus(`🤖 正在載入 WebGPU AI 模型 (${modelName})...`);
    log(`使用 WebGPU 載入模型：${modelName}`);
    updateProgress(0);

    const transcriber = await pipeline('automatic-speech-recognition', modelName, {
        device: 'webgpu',
        progress_callback: (info) => {
            console.log('模型載入進度:', info);

            if (info.status === 'progress') {
                const percent = Math.round(info.progress || 0);
                updateProgress(percent);
                setStatus(`⏳ 下載模型資源中... ${percent}%`);
                return;
            }

            if (info.status === 'ready') {
                updateProgress(100);
                setStatus('✅ WebGPU 模型已就緒，準備開始辨識。');
                return;
            }

            if (info.status) {
                setStatus(`⏳ 模型載入中：${info.status}`);
            }
        }
    });

    cachedTranscriber = transcriber;
    cachedModelName = modelName;

    log('WebGPU 模型載入完成。');
    return transcriber;
}

async function readCsvFirstColumn(file) {
    const csvContent = await file.text();

    return new Promise((resolve, reject) => {
        Papa.parse(csvContent, {
            skipEmptyLines: true,
            complete: (results) => {
                try {
                    const textList = results.data
                        .map(row => String(row?.[0] ?? '').trim())
                        .filter(t => t.length > 0);

                    resolve(textList);
                } catch (error) {
                    reject(error);
                }
            },
            error: reject
        });
    });
}

async function decodeAudioFile(audioCtx, file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
}

function getAudioFiles(files) {
    return Array.from(files).filter(file => {
        const name = file.name.toLowerCase();
        return name.endsWith('.wav') || name.endsWith('.mp3') || name.endsWith('.m4a') || name.endsWith('.ogg');
    });
}

function makeCsvDownload(matchResults) {
    const finalCsv = Papa.unparse(matchResults);
    const blob = new Blob(['\ufeff' + finalCsv], {
        type: 'text/csv;charset=utf-8;'
    });
    const url = URL.createObjectURL(blob);

    resultSection.innerHTML = `
        <a href="${url}" download="Result.csv" class="btn-success">
            ⬇️ 下載輸出結果 CSV
        </a>
    `;
}

startBtn.addEventListener('click', async () => {
    const isModeB = modeB.checked;
    const selectedModel = modelSelect.value;
    const modelName = resolveModelName(selectedModel);

    if (isModeB && !csvFileInput.files[0]) {
        alert('模式 B 需要上傳文本 CSV 供比對！');
        return;
    }

    if (folderFileInput.files.length === 0) {
        alert('請選擇包含音檔的語音資料夾！');
        return;
    }

    startBtn.disabled = true;
    resultSection.innerHTML = '';
    statusLog.innerHTML = '';
    updateProgress(0);

    let textList = [];
    const matchResults = [];

    try {
        log('目前加速模式：WebGPU');
        log(`使用模型：${modelName}`);

        if (isModeB) {
            setStatus('📄 正在解析文本資料...');
            textList = await readCsvFirstColumn(csvFileInput.files[0]);

            if (textList.length === 0) {
                throw new Error('CSV 第一欄沒有讀到可用文本。');
            }

            log(`讀取到 ${textList.length} 句文本供比對。`);
        }

        const transcriber = await createWebGPUTranscriber(modelName);

        setStatus('📁 讀取資料夾音檔...');
        const audioFiles = getAudioFiles(folderFileInput.files);

        if (audioFiles.length === 0) {
            throw new Error('資料夾內沒有找到有效音檔。支援 wav / mp3 / m4a / ogg。');
        }

        log(`找到 ${audioFiles.length} 個音檔。`);

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        for (let i = 0; i < audioFiles.length; i++) {
            const file = audioFiles[i];
            const fileName = file.name;

            setStatus(`🎧 WebGPU 辨識中 (${i + 1}/${audioFiles.length})：${fileName}`);
            log(`處理中：${fileName}`);

            const float32Data = await decodeAudioFile(audioCtx, file);

            const whisperRes = await transcriber(float32Data, {
                language: 'chinese',
                task: 'transcribe'
            });

            let heardText = String(whisperRes.text || '').trim();
            heardText = converter(heardText);

            if (isModeB) {
                const match = stringSimilarity.findBestMatch(heardText, textList);
                const score = Math.round(match.bestMatch.rating * 100);

                matchResults.push({
                    '匹配文本': match.bestMatch.target,
                    '語音檔名': fileName,
                    '辨識內容': heardText,
                    '置信度': `${score}%`,
                    '加速模式': 'WebGPU',
                    '模型': modelName
                });
            } else {
                matchResults.push({
                    '語音檔名': fileName,
                    '辨識內容 (純繁體)': heardText,
                    '加速模式': 'WebGPU',
                    '模型': modelName
                });
            }

            updateProgress(((i + 1) / audioFiles.length) * 100);
        }

        setStatus('🎉 任務完成！');
        log('全部音檔處理完成。');
        makeCsvDownload(matchResults);

    } catch (err) {
        console.error(err);
        setStatus('❌ 發生錯誤');
        log(err.message || String(err));
        log('此版本為 WebGPU 專用版，不會改用 WASM。');
    } finally {
        startBtn.disabled = false;
    }
});
