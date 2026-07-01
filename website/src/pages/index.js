import React from 'react';
import {Redirect} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// The site root has no standalone landing page: visiting
// https://gnust-company.github.io/daily_papers_digest/ sends the reader
// straight to the newest digest (route computed at build time in
// docusaurus.config.js and exposed via customFields.latestDigestRoute).
// To restore a real homepage later (CV / blog / project showcase), replace
// this redirect with page content again.
export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const latestDigestRoute =
    siteConfig.customFields?.latestDigestRoute || '/docs';
  return <Redirect to={latestDigestRoute} />;
}
