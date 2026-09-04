/**
 * IndexedDB storage utility for large media files (Images, Videos, Audio).
 * Bypasses the 5MB browser sessionStorage quota limit by storing raw Blobs/Files in IndexedDB.
 */

const DB_NAME = "deepguard_media_db";
const DB_VERSION = 1;
const STORE_NAME = "media_files";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined" || !window.indexedDB) {
      reject(new Error("IndexedDB is not supported in this environment"));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "scanId" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Store a raw media File in IndexedDB
 */
export async function storeScanFile(scanId: string, file: File): Promise<void> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);

      const record = {
        scanId,
        file,
        name: file.name,
        type: file.type,
        size: file.size,
        storedAt: Date.now(),
      };

      const req = store.put(record);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("Failed to store file in IndexedDB:", err);
  }
}

/**
 * Retrieve a stored media File from IndexedDB
 */
export async function getScanFile(scanId: string): Promise<File | null> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(scanId);

      req.onsuccess = () => {
        const result = req.result;
        if (!result) {
          resolve(null);
          return;
        }

        // If stored as native File
        if (result.file instanceof File) {
          resolve(result.file);
        } else if (result.file instanceof Blob) {
          resolve(new File([result.file], result.name || "media", { type: result.type }));
        } else {
          resolve(null);
        }
      };

      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    console.warn("Failed to retrieve file from IndexedDB:", err);
    return null;
  }
}

/**
 * Clean up old scans from IndexedDB (keep last 10 scans)
 */
export async function pruneOldScanFiles(keepMax: number = 10): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();

    req.onsuccess = () => {
      const all = req.result;
      if (all && all.length > keepMax) {
        all.sort((a, b) => (a.storedAt || 0) - (b.storedAt || 0));
        const toDelete = all.slice(0, all.length - keepMax);
        for (const item of toDelete) {
          store.delete(item.scanId);
        }
      }
    };
  } catch {
    // Ignore cleanup errors
  }
}
