/** Canonical metadata shared by pages, feeds and deployment configuration. */
export const SITE_URL = 'https://tilelang-tutorial.huandu.me';

export const SITE = {
  title: 'TileLang 系统教程',
  shortName: 'TileLang Tutorial',
  description:
    '从设计思想、算子实战到编译期、NVIDIA 与昇腾后端，一部可交互的 TileLang 中文系统教程。',
  author: 'Huan Du',
  email: 'i@huandu.me',
  blog: 'https://huandu.me',
  github: 'https://github.com/huandu',
  repository: 'https://github.com/huandu/tilelang-tutorial',
  domain: 'tilelang-tutorial.huandu.me',
  locale: 'zh-CN',
} as const;

export interface NavLink {
  href: string;
  label: string;
  external?: boolean;
}

export const NAV_ITEMS: readonly NavLink[] = [
  { href: '/', label: '全部章节' },
  { href: '/notes/', label: '编者笔记' },
  { href: '/references/', label: '资料与版本' },
  { href: SITE.blog, label: '博客', external: true },
];
