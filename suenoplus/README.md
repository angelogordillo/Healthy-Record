# SueñoPlus – Bitácora de sueño

Herramienta interactiva del módulo de sueño de **Healthy Record**.

Permite a los colaboradores:

- Registrar diariamente horas y calidad de sueño
- Recibir sugerencias personalizadas según su historial
- Marcar qué recomendaciones cumplieron
- Ver el impacto de las sugerencias en su descanso
- Configurar recordatorios (hora de dormir, wind-down, corte de cafeína)
- Seguir su racha y estadísticas

## Cómo usarla

1. Abre `index.html` en cualquier navegador (móvil o escritorio).
2. Todo se guarda localmente en el dispositivo del usuario (localStorage).
3. No requiere cuenta ni conexión a internet después de cargar.

## Cómo publicarla

### Opción A – Ruta dentro del sitio
Simplemente deja esta carpeta en el repositorio.  
La app quedará disponible en:

```
https://healthyrecord.org/suenoplus/
```

### Opción B – Subdominio (recomendado)
1. Configura el subdominio `sueno.healthyrecord.org` apuntando al deploy.
2. Sirve el contenido de esta carpeta como raíz del subdominio.

### Opción C – Embebida con iframe
En la página del módulo de sueño (`/sueño`) puedes usar:

```html
<iframe 
  src="/suenoplus/" 
  width="100%" 
  height="820" 
  style="border:none; border-radius:16px; max-width:480px; margin:0 auto; display:block;"
  title="Bitácora de sueño interactiva">
</iframe>
```

## Notas técnicas

- Aplicación 100% frontend (HTML + CSS + JavaScript)
- Sin dependencias externas (excepto Google Fonts)
- Compatible con PWA (se puede agregar a pantalla de inicio)
- Los datos nunca salen del dispositivo del usuario

---

Hecho para Healthy Record · Módulo de Sueño
