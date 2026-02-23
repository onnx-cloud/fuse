import React from 'react';
import { render } from '@testing-library/react';
import { ChatWidget } from './chatWidget';

test('ChatWidget basic smoke', () => {
  const { container } = render(<ChatWidget />);
  expect(container).toBeTruthy();
});
