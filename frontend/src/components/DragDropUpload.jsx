import { useState, useRef } from 'react';
import { api } from '../api/client';

export default function DragDropUpload({ onFileUploaded }) {
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.dataTransfer.files;
    if (files?.length) {
      uploadFiles(files);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Only dismiss if leaving the overlay entirely
    if (e.currentTarget === e.target) {
      onFileUploaded(null);
    }
  };

  const uploadFiles = async (files) => {
    setUploading(true);
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        const data = await api.uploadFile(formData);
        if (onFileUploaded) {
          onFileUploaded(data);
        }
      }
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files?.length) {
      uploadFiles(e.target.files);
    }
  };

  return (
    <div
      className="drag-drop-zone active"
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onDragLeave={handleDragLeave}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileSelect}
      />
      <span>
        {uploading ? '📤 Uploading...' : '📂 Drop files here to upload'}
      </span>
    </div>
  );
}
