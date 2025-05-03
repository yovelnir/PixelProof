import { render, screen, fireEvent } from '@testing-library/react';
import ImageUpload from '../ImageUpload';

describe('ImageUpload', () => {
  const mockOnUpload = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders upload button with correct text', () => {
    render(<ImageUpload onUpload={mockOnUpload} buttonText="Test Upload" />);
    expect(screen.getByText('Test Upload')).toBeInTheDocument();
  });

  it('shows error for invalid file type', () => {
    render(<ImageUpload onUpload={mockOnUpload} />);
    
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByLabelText(/click to upload/i);
    
    fireEvent.change(input, { target: { files: [file] } });
    
    expect(screen.getByText(/please upload a valid image file/i)).toBeInTheDocument();
  });

  it('shows error for file size > 5MB', () => {
    render(<ImageUpload onUpload={mockOnUpload} />);
    
    const largeFile = new File(['x'.repeat(6 * 1024 * 1024)], 'large.jpg', { type: 'image/jpeg' });
    const input = screen.getByLabelText(/click to upload/i);
    
    fireEvent.change(input, { target: { files: [largeFile] } });
    
    expect(screen.getByText(/image size should be less than 5MB/i)).toBeInTheDocument();
  });

  it('enables submit button after valid file upload', () => {
    render(<ImageUpload onUpload={mockOnUpload} />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByLabelText(/click to upload/i);
    
    fireEvent.change(input, { target: { files: [file] } });
    
    const submitButton = screen.getByRole('button');
    expect(submitButton).not.toBeDisabled();
  });
}); 