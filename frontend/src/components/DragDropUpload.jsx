import { useState, useRef } from 'react';
import { api } from '../api/client';

export default function DragDropUpload({ onFileUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragIn = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragOut = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);

    const files = e.dataTransfer.files;
    if (files?.length) {
      uploadFiles(files);
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
      className={`drag-drop-zone ${dragging ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileSelect}
      />
      {uploading ? (
        <span>📤 Uploading...</span>
      ) : dragging ? (
        <span>📂 Drop files here</span>
      ) : null}
    </div>
  );
}
