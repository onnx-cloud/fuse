import React from 'react';
import { render } from '@testing-library/react';
import { ChatWidget } from './chatWidget';

test('ChatWidget renders', () => {
  const { container } = render(<ChatWidget />);
  expect(container.textContent).toContain('Fuse Copilot');
});
