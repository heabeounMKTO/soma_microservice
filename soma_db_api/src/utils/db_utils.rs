use anyhow::{Error, Result};
use deadpool_postgres::{GenericClient, Manager, Pool};
use dotenvy::dotenv;
use postgres::NoTls;
use std::env;

pub async fn insert_one_face_vector(pool: Pool) -> Result<()> {
    Ok(())
}

/// gets a mf pool , creates tables if it not already there.
pub async fn init_pool() -> std::io::Result<Pool> {
    dotenv().ok();
    let DB_HOST = env::var("DB_HOST").expect("cannot read DB_HOST");
    let DB_PORT = env::var("DB_PORT").expect("cannot read DB_PORT");
    let DB_USER = env::var("DB_USER").expect("cannot read DB_USER");
    let DB_PASSWORD = env::var("DB_PASSWORD").expect("cannot read DB_PASSWORD");
    let DB_DATABASE = env::var("DB_DATABASE").expect("cannot read DB_DATABASE");

    let mut poolcfg = deadpool_postgres::Config::default();
    poolcfg.user = Some(DB_USER);
    poolcfg.password = Some(DB_PASSWORD);
    poolcfg.host = Some(DB_HOST);
    poolcfg.dbname = Some(DB_DATABASE);
    poolcfg.port = Some(DB_PORT.parse::<u16>().unwrap());
    let pool: Pool = poolcfg.create_pool(None, NoTls).unwrap();
    let _get_pool = pool.get().await.unwrap();
    _get_pool
        .batch_execute(
            "
    CREATE TABLE IF NOT EXISTS frame_tags (
                                        id bigserial PRIMARY KEY,
										frame_name varchar(512),
										original_source varchar(512),
										orignial_type int,
                                        face_count int,
										frame_tags varchar(4096));",
        )
        .await
        .unwrap();

    _get_pool
        .batch_execute(
            "
        CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS face_embeddings (id bigserial PRIMARY KEY, 
                                               name varchar(255), 
                                               face_uuid varchar(512) NOT NULL,
                                               gender int ,
                                               embedding vector(512));
",
        )
        .await
        .unwrap();
    Ok(pool)
}
