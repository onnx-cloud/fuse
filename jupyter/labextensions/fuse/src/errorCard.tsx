import React from 'react';

export const ErrorCard = ({ error }: { error: any }) => {
  return React.createElement(
    'div',
    { style: { border: '1px solid #e00', padding: '10px', borderRadius: '6px' } },
    React.createElement('h3', { style: { margin: 0, color: '#a00' } }, 'Fuse Error'),
    React.createElement('pre', { style: { whiteSpace: 'pre-wrap' } }, JSON.stringify(error, null, 2)),
  );
};
