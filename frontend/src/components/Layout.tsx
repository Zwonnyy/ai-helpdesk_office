import { NavLink, Outlet } from 'react-router-dom'

export default function Layout() {
  return (
    <div className='app-shell'>
      <aside className='sidebar'>
        <div className='brand'>
          <span className='brand-mark'>AH</span>
          <div>
            <strong>AI 헬프데스크</strong>
            <span>티켓 라우터</span>
          </div>
        </div>

        <nav aria-label='주요 메뉴'>
          <NavLink to='/' end>
            <span aria-hidden='true'>▦</span> 대시보드
          </NavLink>
          <NavLink to='/tickets/new'>
            <span aria-hidden='true'>＋</span> 새 티켓
          </NavLink>
        </nav>

        <div className='sidebar-foot'>
          <span className='system-dot' />
          담당자 검토 워크플로
        </div>
      </aside>

      <div className='workspace'>
        <header className='topbar'>
          <div>
            <p className='eyebrow'>IT 운영</p>
            <span className='topbar-title'>헬프데스크 콘솔</span>
          </div>
          <NavLink className='button primary compact' to='/tickets/new'>
            + 새 티켓
          </NavLink>
        </header>
        <main className='content'>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
