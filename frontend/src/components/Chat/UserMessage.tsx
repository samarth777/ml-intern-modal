import { useState, useRef, useEffect } from 'react';
import { Box, Stack, Typography, IconButton, Tooltip, TextField } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import CheckIcon from '@mui/icons-material/Check';
import type { UIMessage } from 'ai';
import type { MessageMeta } from '@/types/agent';

interface UserMessageProps {
  message: UIMessage;
  isLastTurn?: boolean;
  onUndoTurn?: () => void;
  onEditAndRegenerate?: (messageId: string, newText: string) => void | Promise<void>;
  isProcessing?: boolean;
}

function extractText(message: UIMessage): string {
  return message.parts
    .filter((p): p is Extract<typeof p, { type: 'text' }> => p.type === 'text')
    .map(p => p.text)
    .join('');
}

export default function UserMessage({
  message,
  isLastTurn = false,
  onUndoTurn,
  onEditAndRegenerate,
  isProcessing = false,
}: UserMessageProps) {
  const showUndo = isLastTurn && !isProcessing && !!onUndoTurn;
  const showEdit = !isProcessing && !!onEditAndRegenerate;
  const text = extractText(message);
  const meta = message.metadata as MessageMeta | undefined;
  const timeStr = meta?.createdAt
    ? new Date(meta.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;

  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(text);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.selectionStart = inputRef.current.value.length;
    }
  }, [isEditing]);

  const handleStartEdit = () => {
    setEditText(text);
    setIsEditing(true);
  };

  const handleConfirmEdit = () => {
    const trimmed = editText.trim();
    if (!trimmed || trimmed === text) {
      setIsEditing(false);
      return;
    }
    setIsEditing(false);
    onEditAndRegenerate?.(message.id, trimmed);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditText(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleConfirmEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  return (
    <Stack
      direction="row"
      spacing={1.5}
      justifyContent="flex-end"
      alignItems="flex-start"
      sx={{
        '& .action-btn': {
          opacity: 0,
          transition: 'opacity 0.15s ease',
        },
        '&:hover .action-btn': {
          opacity: 1,
        },
      }}
    >
      {!isEditing && (showUndo || showEdit) && (
        <Stack className="action-btn" direction="row" spacing={0.25} sx={{ mt: 0.75 }}>
          {showEdit && (
            <Tooltip title="Edit & regenerate" placement="left">
              <IconButton
                onClick={handleStartEdit}
                size="small"
                sx={{
                  width: 24,
                  height: 24,
                  color: 'var(--muted-text)',
                  '&:hover': {
                    color: 'var(--accent-yellow)',
                    bgcolor: 'rgba(255,157,0,0.08)',
                  },
                }}
              >
                <EditIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          )}
          {showUndo && (
            <Tooltip title="Remove this turn" placement="left">
              <IconButton
                onClick={onUndoTurn}
                size="small"
                sx={{
                  width: 24,
                  height: 24,
                  color: 'var(--muted-text)',
                  '&:hover': {
                    color: 'var(--accent-red)',
                    bgcolor: 'rgba(244,67,54,0.08)',
                  },
                }}
              >
                <CloseIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          )}
        </Stack>
      )}

      <Box
        sx={{
          maxWidth: { xs: '88%', md: '72%' },
          bgcolor: 'var(--surface)',
          borderRadius: 1.5,
          borderTopRightRadius: 4,
          px: { xs: 1.5, md: 2.5 },
          py: 1.5,
          border: '1px solid var(--border)',
        }}
      >
        {isEditing ? (
          <Stack spacing={1}>
            <TextField
              inputRef={inputRef}
              multiline
              fullWidth
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={handleKeyDown}
              variant="outlined"
              size="small"
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontFamily: 'inherit',
                  fontSize: '0.925rem',
                  lineHeight: 1.65,
                  color: 'var(--text)',
                  '& fieldset': { borderColor: 'var(--accent-yellow)', borderWidth: 1.5 },
                  '&:hover fieldset': { borderColor: 'var(--accent-yellow)' },
                  '&.Mui-focused fieldset': { borderColor: 'var(--accent-yellow)' },
                },
              }}
            />
            <Stack direction="row" spacing={0.5} justifyContent="flex-end">
              <Tooltip title="Cancel (Esc)">
                <IconButton
                  onClick={handleCancelEdit}
                  size="small"
                  sx={{ color: 'var(--muted-text)', '&:hover': { color: 'var(--accent-red)' } }}
                >
                  <CloseIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Confirm (Enter)">
                <IconButton
                  onClick={handleConfirmEdit}
                  size="small"
                  sx={{ color: 'var(--accent-green)', '&:hover': { bgcolor: 'rgba(47,204,113,0.1)' } }}
                >
                  <CheckIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>
        ) : (
          <Typography
            variant="body1"
            sx={{
              fontSize: '0.925rem',
              lineHeight: 1.65,
              color: 'var(--text)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {text}
          </Typography>
        )}

        {timeStr && !isEditing && (
          <Typography
            variant="caption"
            sx={{ color: 'var(--muted-text)', mt: 0.5, display: 'block', textAlign: 'right', fontSize: '0.7rem' }}
          >
            {timeStr}
          </Typography>
        )}
      </Box>
    </Stack>
  );
}
