import React from 'react';

const Sidebar: React.FC = () => {
  return (
    <aside style={{ width: '200px', background: '#f0f0f0', padding: '1rem', height: 'calc(100vh - 64px)' }}>
      <nav>
        <ul style={{ listStyle: 'none', padding: 0 }}>
          <li style={{ marginBottom: '1rem' }}><a href="#" style={{ textDecoration: 'none', color: '#333' }}>Dashboard</a></li>
          <li style={{ marginBottom: '1rem' }}><a href="#" style={{ textDecoration: 'none', color: '#333' }}>Video Stream</a></li>
          <li style={{ marginBottom: '1rem' }}><a href="#" style={{ textDecoration: 'none', color: '#333' }}>Control</a></li>
          <li style={{ marginBottom: '1rem' }}><a href="#" style={{ textDecoration: 'none', color: '#333' }}>Replay</a></li>
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar;
