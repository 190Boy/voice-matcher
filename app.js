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
const progressNote = document.getElementById('progressNote');
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

function setProgressNote(text, visible = true) {
    if (!progressNote) return;
    progressNote.textContent = text;
    progressNote.style.display = visible ? 'block' : 'none';
}

function createMonotonicModelProgress() {
    const resources = new Map();
    let displayed = 0;

    function getKey(info) {
        return info.file || info.name || info.url || info.key || info.status || '模型資源';
    }

    return function onProgress(info) {
        console.log('模型載入進度:', info);

        if (info.status === 'ready') {
            displayed = 100;
            updateProgress(100);
            setStatus('✅ WebGPU 模型已就緒，準備開始辨識。');
            setProgressNote('模型載入完成。', false);
            return;
        }

        if (info.status === 'progress') {
            const key = getKey(info);

            let current = null;
            let total = null;

            if (typeof info.loaded === 'number' && typeof info.total === 'number' && info.total > 0) {
                current = info.loaded;
                total = info.total;
            } else if (typeof info.progress === 'number') {
                // Transformers.js 常見 progress 是「單一檔案 0~100 進度」，
                // 所以這裡自己做估算總進度，而且不允許倒退。
                current = Math.max(0, Math.min(100, info.progress));
                total = 100;
            }

            if (current !== null && total !== null) {
                const previous = resources.get(key) || { current: 0, total };
                resources.set(key, {
                    current: Math.max(previous.current || 0, current),
                    total: Math.max(previous.total || total, total)
                });

                let sumCurrent = 0;
                let sumTotal = 0;

                for (const item of resources.values()) {
                    sumCurrent += item.current;
                    sumTotal += item.total;
                }

                let estimated = sumTotal > 0 ? Math.round((sumCurrent / sumTotal) * 100) : 0;

                // 進度不允許倒退；最多先顯示到 99，ready 後才 100。
                estimated = Math.min(99, estimated);
                displayed = Math.max(displayed, estimated);

                updateProgress(displayed);
                setStatus(`⏳ 下載模型資源中... ${displayed}%`);
                setProgressNote(`目前檔案：${key}。此為估算總進度，已避免進度倒退。`);
                return;
            }

            displayed = Math.max(displayed, 3);
            updateProgress(displayed);
            setStatus('⏳ 正在準備模型資源...');
            setProgressNote(`目前狀態：${key}。此階段可能無法取得精準百分比。`);
            return;
        }

        if (info.status) {
            setStatus(`⏳ 模型載入中：${info.status}`);
            setProgressNote('模型包含多個檔案與初始化階段，部分階段沒有精準百分比。');
        }
    };
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
        updateProgress(100);
        setProgressNote('模型已在本次頁面中載入過，直接沿用快取。', false);
        return cachedTranscriber;
    }

    cachedTranscriber = null;
    cachedModelName = null;

    setStatus(`🤖 正在載入 WebGPU AI 模型 (${modelName})...`);
    log(`使用 WebGPU 載入模型：${modelName}`);
    updateProgress(0);
    setProgressNote('模型下載包含多個檔案，進度會以估算總進度顯示。');

    const transcriber = await pipeline('automatic-speech-recognition', modelName, {
        device: 'webgpu',
        progress_callback: createMonotonicModelProgress()
    });

    cachedTranscriber = transcriber;
    cachedModelName = modelName;

    setProgressNote('模型載入完成。', false);
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
        if (typeof window.__restoreAudioLabScore === 'function') {
            window.__restoreAudioLabScore();
        }

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
        setProgressNote('模型載入完成，接下來進度代表音檔處理進度。');
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
        setProgressNote('全部音檔處理完成。', false);
        log('全部音檔處理完成。');
        makeCsvDownload(matchResults);

        if (typeof window.__restoreAudioLabScore === 'function') {
            window.__restoreAudioLabScore();
        }

    } catch (err) {
        console.error(err);
        setStatus('❌ 發生錯誤');
        setProgressNote('處理中斷，請查看下方日誌或 F12 Console。');
        log(err.message || String(err));
        log('此版本為 WebGPU 專用版，不會改用 WASM。');

        if (typeof window.__restoreAudioLabScore === 'function') {
            window.__restoreAudioLabScore();
        }
    } finally {
        startBtn.disabled = false;
    }
});
