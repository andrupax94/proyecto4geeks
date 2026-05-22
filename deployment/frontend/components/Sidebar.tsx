"use client";

import Link from "next/link";

export default function Sidebar() {

  return (
    <aside className="sidebar">

      <h2>MIVIA</h2>

      <nav>

        <ul>

          <li>
            <Link href="/">
              Inicio
            </Link>
          </li>

          <li>
            <Link href="/dashboard">
              Dashboard
            </Link>
          </li>

          <li>
            <Link href="/wiki">
              Wiki
            </Link>
          </li>

          <li>
            <Link href="/soundlist">
              SoundList
            </Link>
          </li>

          <li>
            <Link href="/testsound">
              TestSound
            </Link>
          </li>

        </ul>

      </nav>

    </aside>
  );
}