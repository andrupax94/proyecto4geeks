"use client"

import { useEffect } from "react"

export default function Home() {

  useEffect(() => {
    fetch("http://localhost:8000/")
      .then(res => res.json())
      .then(data => console.log(data))
      .catch(err => console.log(err))
  }, [])

  return (
    <div>
      Frontend funcionando
    </div>
  )
}