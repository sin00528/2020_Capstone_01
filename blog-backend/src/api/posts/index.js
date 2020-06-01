import Router from 'koa-router';
import * as postsCtrl from './posts.ctrl';

const posts = new Router();


posts.get('/', postsCtrl.list);
posts.post('/', postsCtrl.write);


const  post = new Router();  // /api/posts/:id
posts.get('/', postsCtrl.read);
posts.delete('/', postsCtrl.remove);
posts.patch('/', postsCtrl.update);

post.use('/:id', postsCtrl.checkObjectId, post.routes());

export default posts;