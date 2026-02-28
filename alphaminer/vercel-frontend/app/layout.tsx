import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Alpha Miners Dashboard',
  description: 'Advanced algorithmic trading strategies dashboard',
  metadataBase: new URL('https://alphaminer-olive.vercel.app'),
  openGraph: {
    title: 'Alpha Miners Dashboard',
    description: 'Advanced algorithmic trading strategies dashboard',
    url: 'https://alphaminer-olive.vercel.app',
    siteName: 'Alpha Miners',
    images: [
      {
        url: 'https://alphaminer-olive.vercel.app/og.png',
        width: 1200,
        height: 630,
        alt: 'Alpha Miners Dashboard'
      }
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Alpha Miners Dashboard',
    description: 'Advanced algorithmic trading strategies dashboard',
    images: ['https://alphaminer-olive.vercel.app/og.png'],
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}