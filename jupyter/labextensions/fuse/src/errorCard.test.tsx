import React from 'react';
import { render } from '@testing-library/react';
import { ErrorCard } from './errorCard';

test('ErrorCard renders JSON content', () => {
  const error = { message: 'boom', friendly: 'Boom' };
  const { container } = render(<ErrorCard error={error} />);
  expect(container.textContent).toContain('Boom');
});
