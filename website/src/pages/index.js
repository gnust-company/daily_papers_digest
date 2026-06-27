import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const latestDigestRoute =
    siteConfig.customFields?.latestDigestRoute || '/docs';
  return (
    <Layout title={`${siteConfig.title}`} description={siteConfig.tagline}>
      <main
        style={{
          maxWidth: 'var(--ifm-container-width)',
          margin: '0 auto',
          padding: '4rem 1rem',
          textAlign: 'center',
        }}>
        <Heading as="h1" style={{fontSize: '2.75rem'}}>
          {siteConfig.title}
        </Heading>
        <p style={{fontSize: '1.3rem', opacity: 0.85}}>{siteConfig.tagline}</p>
        <p style={{opacity: 0.7, marginTop: '1rem'}}>
          Kho lưu trữ các bản tóm tắt bài báo, sắp xếp theo{' '}
          <strong>Năm → Tháng → Ngày</strong>.
        </p>
        <div
          style={{
            display: 'flex',
            gap: '1rem',
            justifyContent: 'center',
            marginTop: '2.5rem',
          }}>
          <Link
            className="button button--primary button--lg"
            to={latestDigestRoute}>
            Xem digest mới nhất
          </Link>
          <Link className="button button--secondary button--lg" to="/blog">
            Blog
          </Link>
        </div>
      </main>
    </Layout>
  );
}
