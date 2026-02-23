import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { AdminWidget } from './adminWidget';

beforeEach(()=>{
  (global as any).fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({}) }));
});

test('AdminWidget renders and can show add form', async ()=>{
  const { getByText } = render(<AdminWidget />);
  const add = getByText('Add Engine');
  fireEvent.click(add);
  expect(getByText('Save')).toBeTruthy();
});

test('Save calls admin endpoint', async ()=>{
  (global as any).fetch = jest.fn((url:string, opts:any) => {
    if(url.endsWith('/fuse/api/llm/admin')) return Promise.resolve({ ok: false });
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
  });
  const { getByText, getByLabelText } = render(<AdminWidget />);
  fireEvent.click(getByText('Add Engine'));
  const nameInput = (document.querySelector('input') as HTMLInputElement);
  fireEvent.change(nameInput, { target: { value: 'test' } });
  fireEvent.click(getByText('Save'));
  await waitFor(()=> expect((global as any).fetch).toHaveBeenCalled());
});